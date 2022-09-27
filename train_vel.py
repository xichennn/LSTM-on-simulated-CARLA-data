## build traffic state as a separate MLP then concatenate with hist traj encodings
# %%
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
traj, pos_x_mean, pos_x_std, pos_y_mean, pos_y_std = data_preprocess.get_traj("Location.csv")

# %%
#hold-out test segment
idx = 0
plt.plot(traj[idx][:,0], traj[idx][:,1],'k', label = f"traj_{idx}")
plt.plot(traj[idx][1000:1100,0], traj[idx][1000:1100,1],'r', label = "prediction")
plt.legend()
plt.show()
# %%
test = traj[idx][1000:1100,:]
np.savetxt("toy_example/test_vel_data.csv", test)

# %%
#forecasting on vid=0
idx = 0
obs_len = 30
target_len = 50
#segment the train and val trajs
train_val_idx = [i for i in range(0,1000)] + [i for i in range(1100,traj[idx].shape[0])]
pos_x, pos_y, frame_start, vids, vel_x, vel_y, ang_z = data_preprocess.windowed_dataset_vel(traj[idx][train_val_idx,:], obs_len+target_len, stride=1)

# %%
#convert to df
df = pd.DataFrame()
df["pos_x"] = pos_x
df["pos_y"] = pos_y
df["frame_start"] = frame_start
df["vid"] = vids
df["vel_x"] = vel_x
df["vel_y"] = vel_y
df["ang_z"] = ang_z

# %%
#remove observed non-moving cars
#TODO: use velocity instead
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

# %%
#translate and rotate traj
data_preprocess.normalize_traj(df1, obs_len)
# %%
## Network Arguments
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

# %%
#split the dataset 8:1:1
n_sample = df1.shape[0]

data_train = pd.concat([df1[0:650], df1[750:]],axis=0)
data_val = df1[650:750]

data_val = data_val.reset_index(drop=True)
data_train = data_train.reset_index(drop=True)

# %%
batch_size = 64
tr = data_preprocess.CarlaDataset_vel(data_train, t_h = obs_len, t_f = target_len)
val = data_preprocess.CarlaDataset_vel(data_val, t_h = obs_len, t_f = target_len)

trDataloader = DataLoader(tr, batch_size = batch_size, shuffle=True, num_workers=2)
valDataloader = DataLoader(val, batch_size = batch_size, shuffle=False, num_workers=2)

# %%
#initialize optimizer
learning_rate = 0.001
net_optimizer = optim.Adam(net.parameters(), lr = learning_rate)
criterion = nn.MSELoss()

# %%
#initialize loss
losses_tr =[]
losses_val = []

if device == "cuda":
    net = nn.DataParallel(net)
    
net.to(device)

# %%
net.load_state_dict(torch.load(f'net_{obs_len}_{target_len}.tar'))

# %%
n_epochs = 30
# %%
# save the best model, initialize only once
# minVal = math.inf
minVal = 1.05

# %%
with trange(n_epochs) as tr:
    
    for it in tr:

        ## Train:________________________________________________________________________________________________________
        net.train()
        batch_loss_tr = 0.
        
        net.zero_grad()
        tr_batch_count = 0.
        for i, data in enumerate(trDataloader):
            '''
            hist_tr:      shape[batch_size, t_h, inp_featurs]
            ts_tr:        shape[batch_size, 1, ts_features]
            fut_tr:       shape[batch_size, t_f, oup_featurs]
            '''
        
            #hist_tr: 
            hist_tr,  ts_tr, fut_tr = data
            #swap 0 and 1 dimension, hist_tr: shape[t_h, batch_size, num_featurs]
            hist_tr, ts_tr, fut_tr = hist_tr.permute(1, 0, 2),  ts_tr.permute(1, 0, 2), fut_tr.permute(1, 0, 2)
            
            hist_tr.to(device)
            ts_tr.to(device)
            fut_tr.to(device)

            #forward pass
            fut_pred = net(hist_tr, ts_tr)
            loss_tr = criterion(fut_pred.to(device), fut_tr.to(device))
            
            #compute the batch loss
            loss_tr.backward()
            batch_loss_tr += loss_tr.item()
            tr_batch_count += 1
            # losses_tr.append(batch_loss_tr/tr_batch_count)
            
            # #backpropagation and update weights
            net_optimizer.step()          
            a = nn.utils.clip_grad_norm_(net.parameters(), 100)

            # if i%100 == 99:
            #     # plot.hist_fut_pred_plot(hist_tr.detach().numpy(), fut_batch_tr.detach().numpy(), fut_pred_tr.detach().numpy(), it, i, mode="train")
            #     #loss for epoch
            #     batch_loss_tr /= 100
            #     losses_tr.append(batch_loss_tr)
                
            #     #progress bar
            #     tr.set_postfix(loss="{0:.3f}".format(batch_loss_tr))
            #     print("Epoch no:",it+1,"| Epoch progress(%):", "| Avg train loss:",format(batch_loss_tr,'0.4f'))
            #     batch_loss_tr = 0.
        
        ## val:___________________________________________________________________________________________________________
        net.eval()

        val_batch_count = 0
        val_loss = 0.
        predictions = {}
        with torch.no_grad():
            for i, data in enumerate(valDataloader):
                #select data
                hist_val, ts_val, fut_val = data
                hist_val, ts_val, fut_val = hist_val.permute(1, 0, 2), ts_val.permute(1, 0 ,2), fut_val.permute(1, 0, 2)
                
                hist_val.to(device)
                ts_val.to(device)
                fut_val.to(device)

                #forward pass
                fut_pred_val = net(hist_val, ts_val)
                loss_val = criterion(fut_pred_val.to(device), fut_val.to(device))


                #compute the loss
                val_loss += loss_val.item()
                val_batch_count += 1
                predictions[i] = fut_pred_val
        # plot.hist_fut_pred_plot(hist_val.detach().numpy(), fut_val.detach().numpy(), fut_pred_val.detach().numpy(), it, b)

        val_loss /= val_batch_count 
        losses_val.append(val_loss) 
        losses_tr.append(batch_loss_tr/tr_batch_count)
        # if it%5 == 4:
        print("Epoch no:",it+1,"| Epoch progress(%):", "| Avg train loss:",format(batch_loss_tr/tr_batch_count,'0.8f'), "| Avg validation loss:",format(val_loss,'0.8f'))
        
        # if val_loss < minVal:
        #     minVal = val_loss
        #     torch.save(net.state_dict(), f'net_{obs_len}_{target_len}.tar')
              # print("Epoch", it+1, "complete.")

        # if batch_loss_tr/tr_batch_count < minVal:
        #     minVal = batch_loss_tr/tr_batch_count
        #     torch.save(net.state_dict(), f'net_{obs_len}_{target_len}.tar')
            







# %%
#plot on val data
decoder_outputs1 = fut_pred_val.to(device).detach().numpy()
fut_val1 = fut_val.to(device).detach().numpy()
hist_val1 = hist_val.to(device).detach().numpy()
plt.plot(hist_val1[:,:,0], hist_val1[:,:,1],color='0.7',label="hist")
plt.plot(decoder_outputs1[:,:,0], decoder_outputs1[:,:,1],'r',label="fut")
plt.plot(fut_val1[:,:,0], fut_val1[:,:,1],'k',label="gt")
plt.ylim((-4,3))
plt.show()

# %%
#check train data
decoder_tr = fut_pred.to(device).detach().numpy()
fut_tr1 = fut_tr.to(device).detach().numpy()
hist_tr1 = hist_tr.to(device).detach().numpy()
plt.plot(hist_tr1[:,:,0], hist_tr1[:,:,1],color='0.7')
plt.plot(decoder_tr[:,:,0], decoder_tr[:,:,1],'r')
plt.plot(fut_tr1[:,:,0], fut_tr1[:,:,1],'k')
plt.plot(hist_tr1[:,1,0], hist_tr1[:,1,1],color='0.7',label="hist")
plt.plot(decoder_tr[:,1,0], decoder_tr[:,1,1],'r',label="fut_pred")
plt.plot(fut_tr1[:,1,0], fut_tr1[:,1,1],'k',label="fut_gt")
plt.legend()
# plt.ylim((-4,3))
plt.show()
# %%
criterion(decoder_outputs_val.to(device), fut_val.to(device))