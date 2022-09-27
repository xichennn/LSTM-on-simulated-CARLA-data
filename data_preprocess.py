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


class CarlaDataset_vel(Dataset):


    def __init__(self, data_df, t_h=30, t_f=50):
        self.T = data_df
        self.t_h = t_h  # length of track history
        self.t_f = t_f  # length of predicted trajectory


    def __len__(self):
        return self.T.shape[0]


    def __getitem__(self,idx):

        # Get track history 'hist' = ndarray[t_h,2], and future track 'fut' = ndarray[t_f,2]
        xy_seq = np.stack(((self.T["pos_x"].values)[idx], (self.T["pos_y"].values)[idx]), axis=-1)
        m = (self.T["translation"].values)[idx]
        # First apply translation
        ls = LineString(xy_seq)
        ls_offset = affine_transform(ls, m)
        xy_translation = np.array(ls_offset.coords[:], dtype='float32')
        hist, fut = xy_translation[:self.t_h,], xy_translation[self.t_h:,]
        # vel_x = self.T["vel_x"][idx][:self.t_h].reshape(-1,1)
        # vel_y = self.T["vel_y"][idx][:self.t_h].reshape(-1,1)
        # ang_z = self.T["ang_z"][idx][:self.t_h].reshape(-1,1)
        vel_x = self.T["vel_x"][idx][self.t_h-1].reshape(-1,1)
        vel_y = self.T["vel_y"][idx][self.t_h-1].reshape(-1,1)
        ang_z = self.T["ang_z"][idx][self.t_h-1].reshape(-1,1)
        # hist_vel = np.concatenate((hist, vel_x, vel_y, ang_z), axis=-1)
        ts = np.concatenate((vel_x, vel_y, ang_z), dtype = 'float32', axis=-1)
        return hist, ts, fut



class CarlaDataset2(Dataset):


    def __init__(self, data, t_h=20, t_f=30):
        self.T = data
        self.t_h = t_h  # length of track history
        self.t_f = t_f  # length of predicted trajectory


    def __len__(self):
        return self.T.shape[0]


    def __getitem__(self,idx):

        # Get track history 'hist' = ndarray[16,2], and future track 'fut' = ndarray[25,2]
        hist, fut = self.T[idx][:self.t_h,], self.T[idx][self.t_h:,]

        return hist, fut

def windowed_dataset(position, input_window = 16, output_window = 25, stride = 1, num_features = 2):
  
    '''
    create a windowed dataset

    : param pos_x:            vehicle position x in collected frames, m*2          
    : param pos_y:            vehicle position y in collected frames
    : param input_window:     number of history samples to give model 
    : param output_window:    number of future samples to predict  
    : param stide:            spacing between windows (2 in this case)
    : return X, Y:            arrays with correct dimensions for LSTM
    :                         (i.e., [input/output window size # examples, # features])
    '''
  
    L = position.shape[0]
    num_samples = (L - input_window - output_window) // stride + 1

    hist = np.zeros([input_window, num_samples, num_features])
    fut = np.zeros([output_window, num_samples, num_features])    
    
    for ff in range(num_features):
        for ii in range(num_samples):
            start_hist = stride * ii
            end_hist = start_hist + input_window
            hist[:, ii, ff] = position[start_hist:end_hist, ff]

            start_fut = stride * ii + input_window
            end_fut = start_fut + output_window 
            fut[:, ii, ff] = position[start_fut:end_fut, ff]

    return hist, fut
def get_traj(file_name):
    '''
    Arguments:
    filename:     the csv file which stores raw data

    Returns:
    traj:         list[ndarray], traj data of each vehicle
    '''
    data_raw = pd.read_csv(file_name, header=None)
    header = ["frame","time","vid","type_id","position_x","position_y","position_z","rotation_x","rotation_y","rotation_z","vel_x","vel_y","angular_z"]
    map = {idx:header[idx] for idx in range(13)}
    data_raw = data_raw.rename(columns = map) 
    #normalize pos_x and pos_y
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
      vel_x = data_raw[data_raw["vid"]==vid[v]]["vel_x"].values.reshape(-1,1)
      vel_y = data_raw[data_raw["vid"]==vid[v]]["vel_y"].values.reshape(-1,1)
      ang_z = data_raw[data_raw["vid"]==vid[v]]["angular_z"].values.reshape(-1,1)
      f = np.array([frame_map[val] for val in data_raw[data_raw["vid"]==vid[v]]["frame"].values]).reshape(-1,1) 
      vi = data_raw[data_raw["vid"]==vid[v]]["vid"].values.reshape(-1,1)
      traj_v =np.concatenate((x, y, f, vi, vel_x, vel_y, ang_z), dtype = 'float32', axis=1)
      #sort by frame number
      traj_v[traj_v[:, 2].argsort()]
      traj[v] = traj_v
    return traj, pos_x_mean, pos_x_std, pos_y_mean, pos_y_std

def windowed_dataset2(traj, window_size, stride):
    '''
    segment the traj into window_size=obs_len+target_len
    Arguments:
    traj:           ndarray, single traj
    return:
    x:              list[ndarray(window_size,)]
    y:              list[ndarray(window_size,)]
    frame_start:    list[]
    vid:            list[]

    '''
    num_samples = (traj.shape[0] - window_size) // stride + 1
    frame_start = [0] * num_samples
    vid = [0] * num_samples
    x = [0] * num_samples
    y = [0] * num_samples
    for i in range(num_samples):
      x[i] = traj[i*stride:i*stride+window_size, 0]
      y[i] = traj[i*stride:i*stride+window_size, 1]
      frame_start[i] = traj[i, 2]
      vid[i] = traj[i, 3]
    return x, y, frame_start, vid

def windowed_dataset_vel(traj, window_size, stride):
    '''
    segment the traj into window_size=obs_len+target_len
    Arguments:
    traj:           ndarray, single traj
    return:
    x:              list[ndarray(window_size,)]
    y:              list[ndarray(window_size,)]
    frame_start:    list[]
    vid:            list[]

    '''
    num_samples = (traj.shape[0] - window_size) // stride + 1
    frame_start = [0] * num_samples
    vid = [0] * num_samples
    x = [0] * num_samples
    y = [0] * num_samples
    vel_x = [0] * num_samples
    vel_y = [0] * num_samples
    ang_z = [0] * num_samples

    for i in range(num_samples):
      x[i] = traj[i*stride:i*stride+window_size, 0]
      y[i] = traj[i*stride:i*stride+window_size, 1]
      frame_start[i] = traj[i, 2]
      vid[i] = traj[i, 3]
      vel_x[i] = traj[i*stride:i*stride+window_size, 4]
      vel_y[i] = traj[i*stride:i*stride+window_size, 5]
      ang_z[i] = traj[i*stride:i*stride+window_size, 6]

    return x, y, frame_start, vid, vel_x, vel_y, ang_z

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

def numpy_to_torch(Xtrain, Xtest, Ytrain, Ytest):
    '''
    convert numpy array to PyTorch tensor
    
    : param Xtrain:                           windowed training input data (input window size, # examples, # features); np.array
    : param Ytrain:                           windowed training target data (output window size, # examples, # features); np.array
    : param Xtest:                            windowed test input data (input window size, # examples, # features); np.array
    : param Ytest:                            windowed test target data (output window size, # examples, # features); np.array
    : return X_train_torch, Y_train_torch,
    :        X_test_torch, Y_test_torch:      all input np.arrays converted to PyTorch tensors 
    '''
    
    X_train_torch = torch.from_numpy(Xtrain).type(torch.Tensor)
    Y_train_torch = torch.from_numpy(Ytrain).type(torch.Tensor)

    X_test_torch = torch.from_numpy(Xtest).type(torch.Tensor)
    Y_test_torch = torch.from_numpy(Ytest).type(torch.Tensor)
    
    return X_train_torch, X_test_torch, Y_train_torch, Y_test_torch

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