# %%
import sys
sys.path.append('/Users/xichen/Documents/paper2-traj-pred/carla-data')
import torch 
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import numpy as np
from tqdm import trange
import random
import model_vel as model
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import data_preprocess

from shapely.geometry import Point, Polygon, LineString, LinearRing
from shapely.affinity import affine_transform, rotate
import math

# %%
#load hold-out traj segment
data_test = np.loadtxt("toy_example/test_vel_data.csv")

# %%
#process the data as training
obs_len = 30
target_len = 50
#segment the test trajs
pos_x, pos_y, frame_start, vids, vel_x, vel_y, ang_z = data_preprocess.windowed_dataset_vel(data_test, obs_len+target_len, stride=1)

#convert to df
df = pd.DataFrame()
df["pos_x"] = pos_x
df["pos_y"] = pos_y
df["frame_start"] = frame_start
df["vid"] = vids
df["vel_x"] = vel_x
df["vel_y"] = vel_y
df["ang_z"] = ang_z

#remove observed non-moving cars
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
df_test = df.drop(non_move).reset_index(drop=True)

#translate and rotate traj
data_preprocess.normalize_traj(df_test, obs_len)
# %%
args = {}
args['input_size'] = 2
args['output_size'] = 2
args['hidden_size_enc'] = 64
args['hidden_size_dec'] = 64
args['num_layers'] = 1
args["embedding_size"] = 8
args['dyn_embedding_size'] = 32
args['use_cuda'] = True
args['obs_len'] = obs_len
args['target_len'] = target_len
args["ts_feature"] = 3 #vel_x, vel_y, ang_z
args["ts_embedding_size"] = 8

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")
if device == "cuda":
    print(f"Using all ({torch.cuda.device_count()}) GPUs...")

# %%
#initialize network
net = model.CarlaNet(args)

n_epochs = 10
criterion = nn.MSELoss()

# %%
data_test_df = df_test
data_test_df = data_test_df.reset_index(drop=True)
# %%
batch_size = 64
test = data_preprocess.CarlaDataset_vel(data_test_df, t_h = obs_len, t_f = target_len)
testDataloader = DataLoader(test, batch_size = batch_size, shuffle=False, num_workers=2)

# %%
if device == "cuda":
    lstm_encoder = nn.DataParallel(net)
net.to(device)

# %%
net.load_state_dict(torch.load(f'net_{obs_len}_{target_len}.tar'))


# %%
## test:___________________________________________________________________________________________________________
net.eval()

test_batch_count = 0
test_loss = 0.
predictions = {}
with torch.no_grad():
    for i, data in enumerate(testDataloader):
        #select data
        hist_test, ts_test, fut_test = data
        hist_test, ts_test, fut_test = hist_test.permute(1, 0, 2), ts_test.permute(1, 0, 2), fut_test.permute(1, 0, 2)
        
        hist_test.to(device)
        ts_test.to(device)
        fut_test.to(device)

        #forward pass
        fut_pred_test = net(hist_test, ts_test)
        loss_test = criterion(fut_pred_test.to(device), fut_test.to(device))

        test_loss += loss_test.item()
        test_batch_count += 1
        predictions[i] = fut_pred_test

test_loss /= test_batch_count 
print("Test loss: ", test_loss)

# %%
#plot on test data
decoder_outputs1 = fut_pred_test.to(device).detach().numpy()
fut_val1 = fut_test.to(device).detach().numpy()
hist_val1 = hist_test.to(device).detach().numpy()
plt.plot(hist_val1[:,:,0], hist_val1[:,:,1],color='0.7')
plt.plot(decoder_outputs1[:,:,0], decoder_outputs1[:,:,1],'r')
plt.plot(fut_val1[:,:,0], fut_val1[:,:,1],'k')
# plt.ylim((-4,3))
plt.show()
# %%
###calculate MSE on each timestamp
save_dir = "figs"
batch_size = 1
mse_val = 0
counts = 0
for sample_at in range(21):

    pred = predictions[sample_at].to(device).detach().numpy()
    translation = list(df_test["translation"][sample_at*batch_size : sample_at*batch_size + batch_size])
    rotation = list(df_test["rotation"][sample_at*batch_size : sample_at*batch_size + batch_size])

    for i in range(batch_size):
        abs_pred = []
        to_try = pred[:,i,:]
        ls = LineString(to_try)
        ls_rotate = rotate(ls, -rotation[i], origin=(0,0))
        M_inv = [1,0,0,1,-translation[i][4], -translation[i][5]]
        ls_offset = affine_transform(ls_rotate, M_inv).coords[:]
        abs_pred.append(ls_offset)
        abs_pred = np.array(abs_pred)
        gt_val = np.stack((df_test["pos_x"].values[sample_at*batch_size+i].reshape(-1,1), df_test["pos_y"].values[sample_at*batch_size+i].reshape(-1,1)), axis = 1)
        # # for j in range(len(traj)):
        # #     plt.plot(traj[j][:,0], traj[j][:,1], color='0.7', linewidth=3, zorder=1)
        # plt.scatter(gt_val[obs_len:,0], gt_val[obs_len:,1],s=10., c='g', label = "future", zorder=2)
        # plt.scatter(abs_pred[:,:,0].flatten(), abs_pred[:,:,1].flatten(), s=10., c='r', label = "pred", zorder=2)
        # # plt.xlim((-102,-88))
        # # plt.ylim((-125, -105))
        # plt.xticks([])
        # plt.yticks([])
        # # plt.legend(loc='upper right')
        # plt.savefig(f"{save_dir}/{550+sample_at*batch_size+i}.png")
        # plt.close('all')
        mse_val += ((abs_pred[0] - gt_val[obs_len:,].squeeze())**2).sum(axis=1)
        counts += 1

# %%
mse_val /= counts
plt.plot(mse_val)

# %%
plt.plot(mse_val, label="obs4_fut5")
plt.legend()
plt.savefig("toy_example/results/figs/obs4_fut5.jpg")

# %%
np.savetxt("toy_example/results/obs4_fut5.csv", mse_val)
# %%
obs3_fut5 = np.loadtxt("toy_example/results/obs3_fut5.csv")
obs4_fut5 = np.loadtxt("toy_example/results/obs4_fut5.csv")
obs5_fut5 = np.loadtxt("toy_example/results/obs5_fut5.csv")
ax = plt.axes()
plt.plot(obs3_fut5,label="3s_hist")
plt.plot(obs4_fut5,label="4s_hist")
plt.plot(obs5_fut5,label="5s_hist")
plt.legend()
ax.set_xticks([10, 20, 30, 40, 50])
ax.set_xticklabels(["1", "2", "3","4","5"])
  
plt.xlabel("Time (s)")
plt.ylabel("MSE (meter)")
plt.show()
# %%
