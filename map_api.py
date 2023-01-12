import logging
import os
import xml.etree.ElementTree as ET
from typing import Any, Dict, List, Mapping, MutableMapping, Optional, Tuple, Union, cast

import numpy as np
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


_PathLike = Union[str, "os.PathLike[str]"]


class Node:
    """
    e.g. a point of interest, or a constituent point of a
    line feature such as a road
    """

    def __init__(self, id: int, x: float, y: float, height: Optional[float] = None):
        """
        Args:
            id: representing unique node ID
            x: x-coordinate in city reference system
            y: y-coordinate in city reference system

        Returns:
            None
        """
        self.id = id
        self.x = x
        self.y = y
        self.height = height

class LaneSegment:
    def __init__(
        self,
        id: int,
        l_neighbor_id: Optional[int],
        r_neighbor_id: Optional[int],
        centerline: np.ndarray,
    ) -> None:
        """Initialize the lane segment.

        Args:
            id: Unique lane ID that serves as identifier for this "Way"
            l_neighbor_id: Unique ID for left neighbor
            r_neighbor_id: Unique ID for right neighbor
            centerline: The coordinates of the lane segment's center line.
        """
        self.id = id
        self.l_neighbor_id = l_neighbor_id
        self.r_neighbor_id = r_neighbor_id
        self.centerline = centerline

class Road:
    def __init__(
        self,
        id: int,
        l_bound: np.ndarray,
        r_bound: np.ndarray,
    ) -> None:
        """Initialize the lane segment.

        Args:
            id: Unique lane ID that serves as identifier for this "Way"
            l_neighbor_id: Unique ID for left neighbor
            r_neighbor_id: Unique ID for right neighbor
            centerline: The coordinates of the lane segment's center line.
        """
        self.id = id
        self.l_bound = l_bound
        self.r_bound = r_bound

def convert_node_id_list_to_xy(node_id_list: List[int], all_graph_nodes: Mapping[int, Node]) -> np.ndarray:
    """
    convert node id list to centerline xy coordinate

    Args:
       node_id_list: list of node_id's
       all_graph_nodes: dictionary mapping node_ids to Node

    Returns:
       centerline
    """
    num_nodes = len(node_id_list)

    if all_graph_nodes[node_id_list[0]].height is not None:
        centerline = np.zeros((num_nodes, 3))
    else:
        centerline = np.zeros((num_nodes, 2))
    for i, node_id in enumerate(node_id_list):
        if all_graph_nodes[node_id].height is not None:
            centerline[i] = np.array(
                [
                    all_graph_nodes[node_id].x,
                    all_graph_nodes[node_id].y,
                    all_graph_nodes[node_id].height,
                ]
            )
        else:
            centerline[i] = np.array([all_graph_nodes[node_id].x, all_graph_nodes[node_id].y])

    return centerline

def convert_dictionary_to_lane_segment_obj(lane_id: int, lane_dictionary: Mapping[str, Any]) -> LaneSegment:
    """
    Not all lanes have predecessors and successors.

    Args:
       lane_id: representing unique lane ID
       lane_dictionary: dictionary with LaneSegment attributes, not yet in object instance form

    Returns:
       ls: LaneSegment object
    """

    l_neighbor_id = None 
    r_neighbor_id = None 
    ls = LaneSegment(
        lane_id,
        l_neighbor_id,
        r_neighbor_id,
        lane_dictionary["centerline"],
    )
    return ls

def append_unique_key_value_pair(lane_obj: MutableMapping[str, Any], way_field: List[Tuple[str, str]]) -> None:
    """
    For the following types of Way "tags", the key, value pair is defined only once within
    the object:
        - has_traffic_control, turn_direction, is_intersection, l_neighbor_id, r_neighbor_id

    Args:
       lane_obj: lane object
       way_field: key and value pair to append

    Returns:
       None
    """
    assert len(way_field) == 2
    k = way_field[0][1]
    v = way_field[1][1]
    lane_obj[k] = v


def extract_node_waypt(way_field: List[Tuple[str, str]]) -> int:
    """
    Given a list with a reference node such as [('ref', '0')], extract out the lane ID.

    Args:
       way_field: key and node id pair to extract

    Returns:
       node_id: unique ID for a node waypoint
    """
    key = way_field[0][0]
    node_id = way_field[0][1]
    assert key == "ref"
    return int(node_id)

def get_lane_identifier(child: ET.Element) -> int:
    """
    Fetch lane ID from XML ET.Element.

    Args:
       child: ET.Element with information about Way

    Returns:
       unique lane ID
    """
    return int(child.attrib["id"])

def extract_node_from_ET_element(child: ET.Element) -> Node:
    """
    Given a line of XML, build a node object. The "node_fields" dictionary will hold "id", "x", "y".
    The XML will resemble:

        <node id="0" x="3168.066310258233" y="1674.663991981186" />

    Args:
        child: xml.etree.ElementTree element

    Returns:
        Node object
    """
    node_fields = child.attrib
    node_id = int(node_fields["id"])
    for element in child:
        way_field = cast(List[Tuple[str, str]], list(element.items()))
        key = way_field[0][1]
        if key == "local_x":
            x = float(way_field[1][1])
        elif key == "local_y":
            y = float(way_field[1][1])

    return Node(id=node_id, x=x, y=y)


def extract_lane_segment_from_ET_element(
    child: ET.Element, all_graph_nodes: Mapping[int, Node]
) -> Tuple[LaneSegment, int]:
    """
    We build a lane segment from an XML element. A lane segment is equivalent
    to a "Way" in our XML file. Each Lane Segment has a polyline representing its centerline.
    The relevant XML data might resemble::

        <way lane_id="9604854">
            <tag k="has_traffic_control" v="False" />
            <tag k="turn_direction" v="NONE" />
            <tag k="is_intersection" v="False" />
            <tag k="l_neighbor_id" v="None" />
            <tag k="r_neighbor_id" v="None" />
            <nd ref="0" />
            ...
            <nd ref="9" />
            <tag k="predecessor" v="9608794" />
            ...
            <tag k="predecessor" v="9609147" />
        </way>

    Args:
        child: xml.etree.ElementTree element
        all_graph_nodes

    Returns:
        lane_segment: LaneSegment object
        lane_id
    """
    lane_obj: Dict[str, Any] = {}
    lane_id = get_lane_identifier(child)
    node_id_list: List[int] = []
    for element in child:
        # The cast on the next line is the result of a typeshed bug.  This really is a List and not a ItemsView.
        way_field = cast(List[Tuple[str, str]], list(element.items()))
        field_name = way_field[0][0]
        if field_name == "k":
            key = way_field[0][1]
            append_unique_key_value_pair(lane_obj, way_field)
        else:
            node_id_list.append(extract_node_waypt(way_field))

    lane_obj["centerline"] = convert_node_id_list_to_xy(node_id_list, all_graph_nodes)
    lane_segment = convert_dictionary_to_lane_segment_obj(lane_id, lane_obj)
    return lane_segment, lane_id

def construct_road_from_ET_element(
    child: ET.Element, lane_objs: Mapping[int, LaneSegment]
):
    road_id = int(child.attrib["id"])
    for element in child:
        if element.tag == "member":
            relation_field = cast(List[Tuple[str, str]], list(element.items()))
            if relation_field[2][1] == "right":
                r_bound_idx = int(relation_field[1][1])
            elif relation_field[2][1] == "left":
                l_bound_idx = int(relation_field[1][1])
    l_bound = lane_objs[l_bound_idx].centerline
    r_bound = lane_objs[r_bound_idx].centerline
    road = Road(
        road_id,
        l_bound,
        r_bound
    )
    return road, road_id

def load_lane_segments_from_xml(map_fpath: _PathLike) -> Mapping[int, LaneSegment]:
    """
    Load lane segment object from xml file

    Args:
       map_fpath: path to xml file

    Returns:
       lane_objs: List of LaneSegment objects
    """
    tree = ET.parse(os.fspath(map_fpath))
    root = tree.getroot()

    logger.info(f"Loaded root: {root.tag}")

    all_graph_nodes = {}
    lane_objs = {}
    roads = {}
    # all children are either Nodes or Ways or relations
    for child in root:
        if child.tag == "node":
            node_obj = extract_node_from_ET_element(child)
            all_graph_nodes[node_obj.id] = node_obj
        elif child.tag == "way":
            lane_obj, lane_id = extract_lane_segment_from_ET_element(child, all_graph_nodes)
            lane_objs[lane_id] = lane_obj
        elif child.tag == "relation":
            road, road_id = construct_road_from_ET_element(child, lane_objs)
            roads[road_id] = road
        else:
            logger.error("Unknown XML item encountered.")
            raise ValueError("Unknown XML item encountered.")
    return roads