from __future__ import print_function, division
from torch.utils.data import Dataset
import scipy.io as scp
import torch
import numpy as np
import random
import pandas as pd

from shapely.geometry import Point, Polygon, LineString, LinearRing
from shapely.affinity import affine_transform, rotate
import math
# random.seed(100)

class CarlaDataset(Dataset):


    def __init__(self, data, t_h=30, t_f=50):
        self.T = data
        self.t_h = t_h  # length of track history
        self.t_f = t_f  # length of predicted trajectory


    def __len__(self):
        return self.T.shape[0]


    def __getitem__(self,idx):

        # Get track history 'hist' = ndarray[16,2], and future track 'fut' = ndarray[25,2]
        hist, fut = self.T[idx][:self.t_h,], self.T[idx][self.t_h:,]

        return hist, fut

class CarlaDataset_vel(Dataset):


    def __init__(self, data_df, t_h=30, t_f=50):
        self.T = data_df
        self.t_h = t_h  # length of track history
        self.t_f = t_f  # length of predicted trajectory


    def __len__(self):
        return self.T.shape[0]


    def __getitem__(self,idx):

        # Get track history 'hist' = ndarray[t_h,2], and future track 'fut' = ndarray[t_f,2]
        hist, fut = self.T["normalized_traj"][idx][:self.t_h,], self.T["normalized_traj"][idx][self.t_h:,]

        #rotate vel_x and vel_y
        vel_x = self.T["vel_x"][idx]
        vel_y = self.T["vel_y"][idx]
        vel_xy_seq = np.stack((vel_x, vel_y), axis=-1)
        m = [1,0,0,1,0,0]
        # First apply translation
        ls = LineString(vel_xy_seq)
        ls_offset = affine_transform(ls, m)
        # Then apply rotation
        angle = self.T["rotation"][idx]
        ls_rotate = rotate(ls_offset, angle, origin=(0,0)).coords[:]
        vel_xy_rot = np.array(ls_rotate, dtype='float32') 
        
        vel_x_rot = vel_xy_rot[:,0][:self.t_h].reshape(-1,1)
        vel_y_rot = vel_xy_rot[:,1][:self.t_h].reshape(-1,1)

        hist_vel = np.concatenate((hist, vel_x_rot, vel_y_rot), axis=-1)

        return hist_vel, fut

def get_traj(file_name):
    '''
    Arguments:
    filename:           the csv file that stores raw data

    Returns:
    traj:               list[ndarray], traj data of each vehicle
                        fields: pos_x, pos_y, vel_x, vel_y, ang_z, vid, frame
    normalizatin params:pos_x_mean, pos_x_std, pos_y_mean, pos_y_std
    '''
    data_raw = pd.read_csv(file_name, header=None)
    header = ["frame","time","vid","type_id","position_x","position_y","position_z","rotation_x","rotation_y","rotation_z","vel_x","vel_y","angular_z"]
    map = {idx:header[idx] for idx in range(13)}
    data_raw = data_raw.rename(columns = map) 
    #normalize pos_x and pos_y to range(-100, 100)
    pos_x_mean = data_raw["position_x"].mean()
    pos_x_std = data_raw["position_x"].std()
    pos_y_mean = data_raw["position_y"].mean()
    pos_y_std = data_raw["position_y"].std()
    data_raw["position_x"] = 100 * (data_raw["position_x"]-pos_x_mean)/pos_x_std
    data_raw["position_y"] = 100 * (data_raw["position_y"]-pos_y_mean)/pos_y_std

    vid = data_raw["vid"].unique()
    frame_map = {data_raw["frame"].unique()[i]: i for i in range(len(data_raw["frame"].unique()))}  
    traj = [0] * len(vid)
    for v in range(len(vid)):
      x = data_raw[data_raw["vid"]==vid[v]]["position_x"].values.reshape(-1,1)
      y = data_raw[data_raw["vid"]==vid[v]]["position_y"].values.reshape(-1,1)
      f = np.array([frame_map[val] for val in data_raw[data_raw["vid"]==vid[v]]["frame"].values]).reshape(-1,1) 
      vi = data_raw[data_raw["vid"]==vid[v]]["vid"].values.reshape(-1,1)
      vel_x = data_raw[data_raw["vid"]==vid[v]]["vel_x"].values.reshape(-1,1)
      vel_y = data_raw[data_raw["vid"]==vid[v]]["vel_y"].values.reshape(-1,1)
      ang_z = data_raw[data_raw["vid"]==vid[v]]["angular_z"].values.reshape(-1,1)
      traj_v =np.concatenate((x, y, vel_x, vel_y, ang_z, vi, f), axis=1)
      traj_v = traj_v.astype('float32')
      #sort by frame number
      traj_v[traj_v[:, -1].argsort()]
      traj[v] = traj_v
    return traj, pos_x_mean, pos_x_std, pos_y_mean, pos_y_std

def windowed_dataset_per_traj(traj, x, y, vel_x, vel_y, ang_z, frame_start, vids, window_size, stride):
    '''
    segment the traj into window_size = obs_len+target_len
    Arguments:
    traj:           ndarray, single traj
    return:
    x:              list[ndarray(window_size,)]
    y:              list[ndarray(window_size,)]
    frame_start:    list[]
    vid:            list[]

    '''
    num_samples = (traj.shape[0] - window_size) // stride + 1
    for i in range(num_samples):
        x.append(traj[i*stride:i*stride+window_size, 0])
        y.append(traj[i*stride:i*stride+window_size, 1])
        vel_x.append(traj[i*stride:i*stride+window_size, 2])
        vel_y.append(traj[i*stride:i*stride+window_size, 3])
        ang_z.append(traj[i*stride:i*stride+window_size, 4])
        vids.append(traj[i, 5])
        frame_start.append(traj[i, 6])
    return x, y, vel_x, vel_y, ang_z, vids, frame_start

def windowed_dataset_all_trajs(trajs, window_size=80, stride=1):
    '''
    segment all vehicles' trajs into window_size = obs_len+target_len and stack together
    Arguments:
    trajs:           list of all trajs

    return:
    df:              dataframe contains fields [pos_x, pos_y, vel_x, vel_y, ang_z, vid, frame_start]              
    '''
    pos_x = []
    pos_y = []
    vel_x = []
    vel_y = []
    ang_z = []
    vids = []
    frame_start=[]
    for idx in range(len(trajs)):
        #segment the trajs
        pos_x, pos_y, vel_x, vel_y, ang_z, vids, frame_start = windowed_dataset_per_traj(
                                        trajs[idx], pos_x, pos_y, vel_x, vel_y, ang_z, frame_start, vids, window_size, stride)
    df = pd.DataFrame()
    df["pos_x"] = pos_x
    df["pos_y"] = pos_y
    df["vel_x"] = vel_x
    df["vel_y"] = vel_y
    df["ang_z"] = ang_z
    df["vid"] = vids
    df["frame_start"] = frame_start

    return df

def remove_static_traj_segments(df, obs_len):
    #TODO: use vel instead
    non_move = []
    for i in range(df.shape[0]):
        x_coord_seq = (df["pos_x"].values)[i]
        y_coord_seq = (df["pos_y"].values)[i]
        xy_seq_hist = np.stack((x_coord_seq[:obs_len], y_coord_seq[:obs_len]), axis=-1)
        xy_seq_fut = np.stack((x_coord_seq[obs_len:], y_coord_seq[obs_len:]), axis=-1)
        ls = LineString(xy_seq_hist)
        ls1 = LineString(xy_seq_fut)
        if ls.length < 0.1 or ls1.length < 0.1:
            non_move.append(i)
    df1 = df.drop(non_move).reset_index(drop=True)
    return df1
  
def normalize_traj(traj_df, obs_len):
    '''
    perform translation and rotation on the traj
    modify traj_df in-place
    '''
    normalized_traj = []
    translation = []
    rotation = []
    for i in range(traj_df.shape[0]):
        xy_seq = np.stack(((traj_df["pos_x"].values)[i], (traj_df["pos_y"].values)[i]), axis=-1)
        start = xy_seq[0]
        
        # First apply translation
        m = [1, 0, 0, 1, -start[0], -start[1]]
        ls = LineString(xy_seq)
        ls_offset = affine_transform(ls, m)
       
        # Now apply rotation, taking care of edge cases
        end = ls_offset.coords[obs_len - 1]
        if end[0] == 0 and end[1] == 0:
            angle = 0.0
        elif end[0] == 0:
            angle = -90.0 if end[1] > 0 else 90.0
        elif end[1] == 0:
            angle = 0.0 if end[0] > 0 else 180.0
        else:
            angle = math.degrees(math.atan(end[1] / end[0]))
            if (end[0] > 0 and end[1] > 0) or (end[0] > 0 and end[1] < 0):
                angle = -angle
            else:
                angle = 180.0 - angle
        # Rotate the trajetory
        ls_rotate = rotate(ls_offset, angle, origin=(0, 0)).coords[:]

        # Normalized trajectory
        norm_xy = np.array(ls_rotate, dtype='float32') 
        # Update the containers
        normalized_traj.append(norm_xy)
        translation.append(m)
        rotation.append(angle)
    traj_df["normalized_traj"] = normalized_traj
    traj_df["translation"] = translation
    traj_df["rotation"] = rotation

def centered_by_vehicle(traj, t_h = 16, t_f = 25, stride = 1, num_features = 2):
    #train x and train y (16 frames and 25 frames)
    hist = []
    fut = []
    # data=[]
    for v in range(len(traj)):
        if len(traj[v][0]) < t_h+t_f:
                continue

    for histstart in range(t_h,len(traj[v][0])-t_f-1, stride):
        histx = traj[v][0][histstart-t_h : histstart]
        histy = traj[v][1][histstart-t_h : histstart]
        futx = traj[v][0][histstart+1 : histstart+t_f+1]
        futy = traj[v][1][histstart+1 : histstart+t_f+1]
        # print(np.array((histx,histy)))
        hist1 = [[histx[i],histy[i]] for i in range(t_h)]
        fut1 = [[futx[j],futy[j]] for j in range(t_f)]

        hist.append(hist1)
        fut.append(fut1)

    hist = np.array(hist,dtype=np.float32)
    fut = np.array(fut,dtype=np.float32)

    data_post = np.array([[[hist[i]],[fut[i]]] for i in range(hist.shape[0])], dtype=object)
    return data_post

def train_test_split(x, y, train_ratio = 0.8):

  '''
  
  split time series into train/test sets
  
  : param x:                      hist of shape [seq_len, num_samples, #features]
  : para y:                       fut of shape [seq_len, num_samples, #features]
  : para split:                   percent of data to include in training set 
  : return t_train, y_train:      time/feature training and test sets;  
  :        t_test, y_test:        (shape: [# samples, 1])
  
  '''
  
  indx_split = int(train_ratio * x.shape[1])
#   indx_train = random.sample(range(0, x.shape[1]), indx_split)
#   indx_test = [x for x in range(0, x.shape[1] )if x not in set(indx_train)]

  indx_train = np.arange(0, indx_split)
  indx_test = np.arange(indx_split, x.shape[1])
  
  x_train = x[:, indx_train, :]
  y_train = y[:, indx_train, :]
  
  x_test = x[:, indx_test, :]
  y_test = y[:, indx_test, :]
  
  return x_train, x_test, y_train, y_test 

def get_preprocessed_trajs(file_path, obs_len, target_len, stride=1):

    trajs, pos_x_mean, pos_x_std, pos_y_mean, pos_y_std = get_traj(file_path)
    df = windowed_dataset_all_trajs(trajs, window_size=obs_len+target_len, stride=stride)
    df1 = remove_static_traj_segments(df, obs_len)
    #translation and rotation in place
    normalize_traj(df1, obs_len)
    # df1["trajs_orig"] = trajs
    df1["pos_x_mean"] = pos_x_mean
    df1["pos_x_std"] = pos_x_std
    df1["pos_y_mean"] = pos_y_mean
    df1["pos_y_std"] = pos_y_std

    return df1

if __name__=="__main__":
    df = get_preprocessed_trajs("Location.csv", obs_len=30, target_len=50, stride=1)

