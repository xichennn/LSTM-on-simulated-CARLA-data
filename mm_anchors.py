# -*- coding: utf-8 -*-

#########################################################################
#we are using the same historic interval but with different len (30,40,50)
#to predict on the same future interval(50). The translation and rotation 
# are now preformed based on the 1st observed point, however, in the case 
# of 30 and 40, they should start from the 20th and 10th point, 
# respectively
#########################################################################
# %%
from toml import TomlDecodeError
import torch 
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import numpy as np
from tqdm import trange
import random
import model_LSTM as model
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import data_preprocess

from shapely.geometry import Point, Polygon, LineString, LinearRing
from shapely.affinity import affine_transform, rotate
import math
import time
from losses import LaplaceNLLLoss, SoftTargetCrossEntropyLoss

from decoders.utils import k_means_anchors, bivariate_gaussian_activation
from metrics.loss_anchor import loss_mm

# %%

class LSTM_anchor(nn.Module):
    def __init__(self, args) -> None:
        '''
        Arguments:
        embedding_size:     Embedding size
        hidden_size:        Hidden size of LSTM
        output_size:        number of features in the output
        '''

        super(LSTM_anchor, self).__init__()
        self.args = args
        self.embedding_size = args["embedding_size"]
        self.hidden_size_enc = args["hidden_size_enc"]
        self.hidden_size_dec = args["hidden_size_dec"]
        self.output_size = args["output_size"]
        self.num_layers = args['num_layers']
        self.num_modes = args['num_modes']
        self.target_len = args["target_len"]

        self.linear1 = nn.Linear(self.output_size, self.embedding_size)
        self.lstm1 = nn.LSTM(self.embedding_size, self.hidden_size_dec, self.num_layers)
        self.linear2 = nn.Linear(self.hidden_size_dec, self.output_size)
        self.prob_op = nn.Linear(self.hidden_size_enc, self.num_modes)
        self.relu = nn.ReLU()
        self.log_softmax = nn.LogSoftmax(dim=1)
        self.dec_lstm = nn.LSTM(self.hidden_size_enc, 128)
        self.op = nn.Linear(128, 5*self.num_modes)


    def generate_anchors(self, tr):
        """
        Function to initialize anchors
        :param tr: train dataset for single agent trajectory prediction
        :return anchors: [num_modes, target_len, 2]
        """

        self.anchors = nn.Parameter(k_means_anchors(self.num_modes, tr)) 

    def forward(self, enc):
        '''
        Params:
        :enc:      final hidden state[0] from LSTM encoder [1, batch_size, hidden_size_enc]
        Return
        :pred: [target_len, num_modes, batch_size, 5]
        :log_probs: [batch_size, num_modes]
  
        '''
        #encoder pass
        log_probs = self.log_softmax(self.prob_op(enc.squeeze(0)))

        enc = enc.repeat(self.target_len, 1, 1)
        h_dec, _ = self.dec_lstm(enc)
        fut_pred = self.op(h_dec)
        fut_pred = fut_pred.view(fut_pred.shape[0], self.num_modes, fut_pred.shape[1], 5)
        fut_pred = bivariate_gaussian_activation(fut_pred)

        anchors = self.anchors.permute(1,0,2).unsqueeze(2)
        fut_pred[...,:2] += anchors

        return fut_pred, log_probs


# %%
#########################################################################
# obs_len_net controls the model input sequence
# args["use_ts"] decides whether to use vel_x and vel_y
#########################################################################
# %%
obs_len = 50
target_len = 50
# %%
trajs, pos_x_mean, pos_x_std, pos_y_mean, pos_y_std = data_preprocess.get_traj("Location.csv")
trajs_sub = []
sub_idx = [0,1,2,4,6,7,8,9,12,13,14,15,16,17]
for i in sub_idx:
    trajs_sub.append(trajs[i])
df = data_preprocess.windowed_dataset_all_trajs(trajs_sub, window_size=obs_len+target_len, stride=1)
df1 = data_preprocess.remove_static_traj_segments(df, obs_len)
#translation and rotation in place
data_preprocess.normalize_traj(df1, obs_len)
# df1["trajs_orig"] = trajs
df1["pos_x_mean"] = pos_x_mean
df1["pos_x_std"] = pos_x_std
df1["pos_y_mean"] = pos_y_mean
df1["pos_y_std"] = pos_y_std

# %%
# visualization.map_traj_vis("./maps/lanelet2/Town03.osm", traj[-2], pos_x_mean, pos_x_std, pos_y_mean, pos_y_std)

# %%

df0 = df1.copy()
train_vid =[398.+ i for i in [1,2,4,6,7,8,9,12,13,14,15,16,17]]

val_test_vid = [398.]
df_train = df0[df0["vid"].isin(train_vid)]
# %%
## hold-out data
df_val = df0[df0["vid"].isin(val_test_vid)][:400]
df_test = df0[df0["vid"].isin(val_test_vid)][400:]
# %%
# plot_idx = 750
# plt.plot((df1["pos_x"].values)[plot_idx], (df1["pos_y"].values)[plot_idx])

# plt.plot((df1["normalized_traj"].values)[plot_idx][:,0], (df1["normalized_traj"].values)[plot_idx][:,1])

# %%
## Network Arguments
obs_len_net = 50
args = {}
args['input_size'] = 2 #2 pos_x, pos_y or 4 if #vel_x, vel_y
args['output_size'] = 2
args['hidden_size_enc'] = 64
args['hidden_size_dec'] = 64
args['num_layers'] = 1
args["embedding_size"] = 8
args['use_cuda'] = True
args['obs_len'] = obs_len_net
args['target_len'] = target_len
args["num_modes"] = 25
args["K"] = 5
args['min_scale'] = 1e-3
args["uncertain"] = True
args['dyn_embedding_size'] = 32
args["ts_embedding_size"] = 8
args["use_ts"] = False
if args["use_ts"]:
    args['input_size'] = 4

args['agg_type'] = 'sample_specific'
args['use_variance'] = False
args['op_len'] = args['target_len']

device = "cuda" if torch.cuda.is_available() else "cpu"
# device = "cpu"
print(f"Using {device} device")
if device == "cuda":
    print(f"Using all ({torch.cuda.device_count()}) GPUs...")

# %%
#split the dataset 8:1:1

n_sample = df_train.shape[0]

if args["use_ts"]:
    data_train = df_train
    data_val = df_val
    data_test = df_test

    data_val = data_val.reset_index(drop=True)
    data_train = data_train.reset_index(drop=True)
    data_test = data_test.reset_index(drop=True)


    train = data_preprocess.CarlaDataset_vel(data_train, t_h=obs_len, t_f=target_len)
    val = data_preprocess.CarlaDataset_vel(data_val, t_h=obs_len, t_f=target_len)
    test = data_preprocess.CarlaDataset_vel(data_test, t_h=obs_len, t_f=target_len)
else:
    data_train = df_train["normalized_traj"]
    data_val = df_val["normalized_traj"]
    data_test = df_test["normalized_traj"]

    data_val = data_val.reset_index(drop=True)
    data_train = data_train.reset_index(drop=True)
    data_test = data_test.reset_index(drop=True)


    train = data_preprocess.CarlaDataset(data_train, t_h=obs_len, t_f=target_len)
    val = data_preprocess.CarlaDataset(data_val, t_h=obs_len, t_f=target_len)
    test = data_preprocess.CarlaDataset(data_test, t_h=obs_len, t_f=target_len)
# %%
batch_size = 64
trDataloader = DataLoader(train, batch_size = batch_size, shuffle=True, num_workers=4)
valDataloader = DataLoader(val, batch_size = batch_size, shuffle=False, num_workers=4)
testDataloader = DataLoader(test, batch_size = batch_size, shuffle=False, num_workers=4)
# plt.plot(val.__getitem__(0)[0][:,0],tr.__getitem__(0)[0][:,1],'k')
# plt.plot(val.__getitem__(0)[1][:,0],tr.__getitem__(0)[1][:,1],'r')
# %%
#initialize network
lstm_encoder = model.lstm_encoder(args)
lstm_decoder = LSTM_anchor(args)

n_epochs = 10
# %%
#initialize optimizer
learning_rate = 0.001
encoder_optimizer = optim.Adam(lstm_encoder.parameters(), lr = learning_rate)
decoder_optimizer = optim.Adam(lstm_decoder.parameters(), lr = learning_rate)
criterion = nn.MSELoss()
reg_loss = LaplaceNLLLoss(reduction="mean")
cls_loss = SoftTargetCrossEntropyLoss(reduction="mean")
# %%
#initialize loss
losses_tr =[]
losses_val = []

if device == "cuda":
    lstm_encoder = nn.DataParallel(lstm_encoder)
    lstm_decoder = nn.DataParallel(lstm_decoder)
lstm_encoder.to(device)
lstm_decoder.to(device)

# %%
# lstm_encoder.load_state_dict(torch.load(f'hist_compr_ts_encoder_{obs_len_net}_{target_len}.tar'))
# lstm_decoder.load_state_dict(torch.load(f'hist_compr_ts_decoder_{obs_len_net}_{target_len}.tar'))

# %%
n_epochs = 2
# %%
# save the best model, initialize only once
minVal = math.inf
# minVal = 1.4
lstm_decoder.module.generate_anchors(train)
anchors = torch.Tensor(lstm_decoder.module.anchors) #[num_modes, target_len, op_dim]

# %%
with trange(n_epochs) as tr:
    
    for it in tr:

        ## Train:________________________________________________________________________________________________________
        lstm_encoder.train()
        lstm_decoder.train()

        batch_loss_tr = 0.
        
        lstm_encoder.zero_grad()
        lstm_decoder.zero_grad()
        tr_batch_count = 0.
        
        for i, data in enumerate(trDataloader):
            '''
            hist_tr:      shape[batch_size, t_h, num_featurs]
            fut_tr:       shape[batch_size, t_f, num_featurs]
            '''
        
            #hist_batch_tr: 
            hist_tr,  fut_tr = data
            hist_tr = hist_tr[:,-obs_len_net:,:]
            #swap 0 and 1 dimension, hist_batch_tr: shape[t_h, batch_size, num_featurs]
            hist_tr, fut_tr = hist_tr.permute(1, 0, 2), fut_tr.permute(1, 0, 2)
            
            hist_tr.to(device)
            fut_tr.to(device)

            #zero the gradients
            encoder_optimizer.zero_grad()
            decoder_optimizer.zero_grad()

            #Initialize loss
            loss_tr = 0.

            # Initialize encoder hidden state
            batch_size = hist_tr.shape[1]
            if device == "cuda":
                    encoder_hidden = (torch.zeros(args["num_layers"], batch_size, args['hidden_size_enc']).to(device),
                                torch.zeros(args["num_layers"], batch_size, args['hidden_size_enc']).to(device)) 
            else:
                encoder_hidden = lstm_encoder.init_hidden(batch_size) 
                                
            #Encode observed trajectory 
            for ei in range(hist_tr.shape[0]):
                encoder_input = hist_tr[ei, :, :].unsqueeze(0)
                encoder_hidden = lstm_encoder(encoder_input, encoder_hidden)

            #Initialize decoder input with the last coordinate in encoder
            # decoder_input = encoder_input[:, :2]
            decoder_input = encoder_input[:,:,:2]
            
            #Initialize decoder hidden state as encoder hidden state
            decoder_hidden = encoder_hidden
            
            #LSTM decoder [target_len, num_modes, batch_size, 5], [batch_size,num_modes]
            pred_res, log_probs = lstm_decoder(decoder_hidden[0])

            loss_reg, loss_cls = loss_mm(pred_res, fut_tr, log_probs, anchors)
            
            loss_tr = torch.mean(loss_reg + loss_cls)
            # loss_tr = loss_tr / args['target_len']
            #compute the batch loss
            loss_tr.backward()
            batch_loss_tr += loss_tr.item()
            tr_batch_count += 1
            # losses_tr.append(batch_loss_tr/tr_batch_count)
            
            # #backpropagation and update weights
            encoder_optimizer.step()
            decoder_optimizer.step()
            
            a = nn.utils.clip_grad_norm_(lstm_encoder.parameters(), 10)
            b = nn.utils.clip_grad_norm_(lstm_decoder.parameters(), 10)
        
        ## val:___________________________________________________________________________________________________________
        lstm_encoder.eval()
        lstm_decoder.eval()
        
        val_batch_count = 0
        val_loss = 0.
        predictions = {}
        with torch.no_grad():
            for i, data in enumerate(valDataloader):
                #select data
                hist_val, fut_val = data
                hist_val = hist_val[:,-obs_len_net:,:]
                hist_val, fut_val = hist_val.permute(1, 0, 2), fut_val.permute(1, 0, 2)
                
                hist_val.to(device)
                fut_val.to(device)
                #Initialize encoder hidden state
                batch_size = hist_val.shape[1]
            
                if device == "cuda":
                    encoder_hidden = (torch.zeros(args["num_layers"], batch_size, args['hidden_size_enc']).to(device),
                                        torch.zeros(args["num_layers"], batch_size, args['hidden_size_enc']).to(device)) 
                else:
                    encoder_hidden = lstm_encoder.init_hidden(batch_size)      
                #Initialize loss
                loss_val = 0.

                #Encode observed trajectory
                for ei in range(hist_val.shape[0]):  
                    encoder_input = hist_val[ei, :, :].unsqueeze(0)
                    encoder_hidden = lstm_encoder(encoder_input, encoder_hidden)

                #Initialize decoder input with the last coordinate in encoder
                decoder_input = encoder_input[:,:,:2]
                #Initialize decoder hidden state as encoder hidden state
                decoder_hidden = encoder_hidden

                pred_res_val, log_probs_val = lstm_decoder(decoder_hidden[0])

                loss_reg_val, loss_cls_val = loss_mm(pred_res_val, fut_val, log_probs_val, anchors)
                
                loss_val = torch.mean(loss_reg_val + loss_cls_val)
        
                #compute the loss
                # loss_val /= args['target_len']
                val_loss += loss_val.item()
                val_batch_count += 1
                predictions[i] = pred_res_val
        # plot.hist_fut_pred_plot(hist_val.detach().numpy(), fut_val.detach().numpy(), fut_pred_val.detach().numpy(), it, b)

        val_loss /= val_batch_count 
        losses_val.append(val_loss) 
        losses_tr.append(batch_loss_tr/tr_batch_count)
        # if it%5 == 4:
        print("Epoch no:",it+1,"| Epoch progress(%):", "| Avg train loss:",format(batch_loss_tr/tr_batch_count,'0.8f'), "| Avg validation loss:",format(val_loss,'0.8f'))
        
        # if val_loss < minVal:
        #     minVal = val_loss
        #     torch.save(lstm_encoder.state_dict(), f'mm_hist_compr_ts_encoder_{obs_len_net}_{target_len}.tar')
        #     torch.save(lstm_decoder.state_dict(), f'mm_hist_compr_ts_decoder_{obs_len_net}_{target_len}.tar')
        # print("Epoch", it+1, "complete.")
        

#%%
          
## test:___________________________________________________________________________________________________________
lstm_encoder.eval()
lstm_decoder.eval()

test_batch_count = 0
test_loss = 0.
predictions_test = {}
predictions_prob_test = {}
losses_test = []

with torch.no_grad():
    for i, data in enumerate(testDataloader):
        #select data
        hist_test, fut_test = data
        hist_test = hist_test[:,-obs_len_net:,:]
        hist_test, fut_test = hist_test.permute(1, 0, 2), fut_test.permute(1, 0, 2)
        
        hist_test.to(device)
        fut_test.to(device)
        #Initialize encoder hidden state
        batch_size = hist_test.shape[1]
    
        if device == "cuda":
            encoder_hidden = (torch.zeros(args["num_layers"], batch_size, args['hidden_size_enc']).to(device),
                                torch.zeros(args["num_layers"], batch_size, args['hidden_size_enc']).to(device)) 
        else:
            encoder_hidden = lstm_encoder.init_hidden(batch_size)      
        #Initialize loss
        loss_test = 0.

        #Encode observed trajectory
        for ei in range(hist_test.shape[0]):  
            encoder_input = hist_test[ei, :, :].unsqueeze(0)
            encoder_hidden = lstm_encoder(encoder_input, encoder_hidden)

        #Initialize decoder input with the last coordinate in encoder
        decoder_input = encoder_input[:,:,:2]
        #Initialize decoder hidden state as encoder hidden state
        decoder_hidden = encoder_hidden

        decoder_outputs_test = torch.zeros(fut_test.shape[0], args["num_modes"], fut_test.shape[1], 2).to(device)
        probs_test = torch.zeros(fut_test.shape[0],  fut_test.shape[1], args["num_modes"],).to(device)
        #decode hidden state in future trajectory
        for di in range(args['target_len']):
            decoder_output, decoder_hidden, log_probs_test = lstm_decoder(decoder_input, decoder_hidden)
            anchor_di = anchors[:,di,:].unsqueeze(1)
            decoder_output = decoder_output.to(device) + anchor_di.to(device)
            decoder_outputs_test[di, :, :, :] = decoder_output
            probs_test[di, :, :] = torch.exp(log_probs_test)
        
            #update loss
            l2_norm_test = torch.linalg.vector_norm(
                decoder_output[:,:,:2].to(device) - fut_test[di,:,:].unsqueeze(0).to(device),
                dim=(-2,-1))
            best_mode_test = l2_norm_test.argmin(dim=0)
            decoder_output_best_test = decoder_output[best_mode_test, :, :2]

            loss_test += criterion(decoder_output_best_test.to(device), fut_test[di, :, :].to(device))

            
            #use own predictions as inputs at next step
            decoder_input = decoder_output_best_test.unsqueeze(0)

        #compute the loss
        loss_test /= args['target_len']
        test_loss += loss_test.item()
        test_batch_count += 1
        predictions_test[i] = decoder_outputs_test
        predictions_prob_test[i] = probs_test

    test_loss /= test_batch_count 
    losses_test.append(test_loss)

    print("test_loss: ", test_loss)

# %%
with open('predcitions_vel_50_50.pickle', 'wb') as handle:
    pickle.dump(predictions, handle, protocol=pickle.HIGHEST_PROTOCOL)

# %%
#plot on val data
decoder_outputs1 = decoder_outputs_val.to("cpu").detach().numpy()
fut_val1 = fut_val.to("cpu").detach().numpy()
hist_val1 = hist_val.to("cpu").detach().numpy()
idx = 14
plt.plot(hist_val1[:,idx,0], hist_val1[:,idx,1],color='0.7')
plt.plot(decoder_outputs1[:,:,idx,0], decoder_outputs1[:,:,idx,1],'r')
plt.plot(fut_val1[:,idx,0], fut_val1[:,idx,1],'k')
# plt.ylim((-4,3))
plt.show()

# %%
# plot on test data
decoder_test = decoder_outputs_test.to("cpu").detach().numpy()
fut_test1 = fut_test.to("cpu").detach().numpy()
hist_test1 = hist_test.to("cpu").detach().numpy()
idx = 8
plt.plot(hist_test1[:,idx,0], hist_test1[:,idx,1],color='0.7')
plt.plot(decoder_test[:,:,idx,0], decoder_test[:,:,idx,1],'r')
plt.plot(fut_test1[:,idx,0], fut_test1[:,idx,1],'k')
plt.ylim((-4,4))
plt.show()

# %%
criterion(decoder_outputs_val.to("cpu"), fut_val.to("cpu"))

# %%
#plot on train data
decoder_outputs2 = decoder_outputs_tr.to("cpu").detach().numpy()
fut_tr = fut_tr.to("cpu").detach().numpy()
hist_tr = hist_tr.to("cpu").detach().numpy()
plt.plot(hist_tr[:,:,0], hist_tr[:,:,1],color='0.7')
plt.plot(decoder_outputs2[:,:,0,0], decoder_outputs2[:,:,0,1],'r')
plt.plot(fut_tr[:,:,0], fut_tr[:,:,1],'k')
plt.show()
# %%
# %%
###calculate MSE on each timestamp
save_dir = "figs"
batch_size = 64
rmse_val = 0
counts = 0
# for sample_at in range(len(predictions_test)):
for sample_at in range(10,12):
    pred = predictions_test[sample_at].to("cpu").detach().numpy()
    pred += df_test["normalized_traj"].values[sample_at*batch_size : sample_at*batch_size + batch_size][0][args["obs_len"]-1,:]
    translation = list(df_test["translation"][sample_at*batch_size : sample_at*batch_size + batch_size])
    rotation = list(df_test["rotation"][sample_at*batch_size : sample_at*batch_size + batch_size])

    for i in range(batch_size):
        abs_pred = np.zeros((args["target_len"], args["num_modes"], args["output_size"]))
        to_try = pred[:,:,i,:] #[target_len, num_modes, batch_size, op_dim]
        for j in range(args["num_modes"]):
            ls = LineString(to_try[:,j,:])
            ls_rotate = rotate(ls, -rotation[i], origin=(0,0))
            M_inv = [1,0,0,1,-translation[i][4], -translation[i][5]]
            ls_offset = affine_transform(ls_rotate, M_inv).coords[:]
            abs_pred[:,j,:] = ls_offset
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
        
        # plt.legend(loc='upper right')
        # plt.savefig(f"{save_dir}/{550+sample_at*batch_size+i}.png")
        # plt.close('all')
        rmse_val += (((abs_pred[0] - gt_val[obs_len:,].squeeze())**2).sum(axis=1))**0.5
        counts += 1
# %%
rmse_val /= counts
plt.plot(rmse_val)

# %%
plt.plot(rmse_val, label=f"obs{obs_len_net}_fut50")
plt.legend()
plt.savefig(f"toy_example/results/figs/obs{obs_len_net}_fut50.jpg")

# %%
np.savetxt(f"toy_example/results/obs{obs_len_net}_fut5.csv", rmse_val)
# %%
obs3_fut5 = np.loadtxt("toy_example/results/obs30_fut5.csv")
obs4_fut5 = np.loadtxt("toy_example/results/obs40_fut5.csv")
obs5_fut5 = np.loadtxt("toy_example/results/obs50_fut5.csv")
ax = plt.axes()
plt.plot(obs3_fut5,label="3s_hist")
plt.plot(obs4_fut5,label="4s_hist")
plt.plot(obs5_fut5,label="5s_hist")
plt.legend()
ax.set_xticks([10, 20, 30, 40, 50])
ax.set_xticklabels(["1", "2", "3","4","5"])
  
plt.xlabel("Time (s)")
plt.ylabel("RMSE (m)")
plt.show()

# %%
###save MSE at each timestamp with ts feature


# %%
rmse_val /= counts
plt.plot(rmse_val)

# %%
plt.plot(rmse_val, label=f"ts_obs{obs_len_net}_fut50")
plt.legend()
plt.savefig(f"toy_example/results/figs/ts_obs{obs_len_net}_fut50.jpg")

# %%
np.savetxt(f"toy_example/results/ts_obs{obs_len_net}_fut5.csv", rmse_val)
# %%
###results comparison
plt.figure(dpi=200)
obs3_fut5 = np.loadtxt("toy_example/results/obs30_fut5.csv")
obs4_fut5 = np.loadtxt("toy_example/results/obs40_fut5.csv")
obs5_fut5 = np.loadtxt("toy_example/results/obs50_fut5.csv")
ts_obs5_fut5 = np.loadtxt("toy_example/results/ts_obs50_fut5.csv")
ax = plt.axes()
plt.plot(obs3_fut5,label="3s_hist")
plt.plot(obs4_fut5,label="4s_hist")
plt.plot(obs5_fut5,label="5s_hist")
plt.plot(ts_obs5_fut5,linestyle="dashed",label="5s_hist_ts")
plt.legend()
ax.set_xticks([10, 20, 30, 40, 50])
ax.set_xticklabels(["1", "2", "3","4","5"])
  
plt.xlabel("Time (s)")
plt.ylabel("RMSE (m)")
# plt.show()
plt.savefig("rmse_compr.jpg")
# %%

predictions_ts = predictions_test
# %%
predictions_obs5 = predictions_test
# %%
predictions_obs4 = predictions_test
# %%
predictions_obs3 = predictions_test

# %%
def denorm(predictions,batch_size=64):
    for sample_at in range(9,10):
        pred= predictions[sample_at].to("cpu").detach().numpy()
        pred += df_test["normalized_traj"].values[sample_at*batch_size : sample_at*batch_size + batch_size][0][args["obs_len"]-1,:]
        translation = list(df_test["translation"][sample_at*batch_size : sample_at*batch_size + batch_size])
        rotation = list(df_test["rotation"][sample_at*batch_size : sample_at*batch_size + batch_size])

        for i in range(47, 48):
            abs_pred = np.zeros((args["target_len"], args["num_modes"], args["output_size"]))
            to_try = pred[:,:,i,:] #[target_len, num_modes, batch_size, op_dim]
            for j in range(args["num_modes"]):
                ls = LineString(to_try[:,j,:])
                ls_rotate = rotate(ls, -rotation[i], origin=(0,0))
                M_inv = [1,0,0,1,-translation[i][4], -translation[i][5]]
                ls_offset = affine_transform(ls_rotate, M_inv).coords[:]
                abs_pred[:,j,:] = ls_offset
            abs_pred = np.array(abs_pred)
            gt_val = np.stack((df_test["pos_x"].values[sample_at*batch_size+i].reshape(-1,1), df_test["pos_y"].values[sample_at*batch_size+i].reshape(-1,1)), axis = 1)
        
    return abs_pred, gt_val
# %%
abs_pred_t, gt_val_t = denorm(predictions_test)


# %%
def pred_map_scale(traj):
    traj= np.array([traj[:,:,0]*pos_x_std/100+pos_x_mean, traj[:,:,1]*pos_y_std/100+pos_y_mean]).T
    return traj
def gt_map_scale(traj):
    traj= np.array([traj[:,0,:]*pos_x_std/100+pos_x_mean, traj[:,1,:]*pos_y_std/100+pos_y_mean]).T
    return traj
abs_pred_t = pred_map_scale(abs_pred_t)
gt_val_t = gt_map_scale(gt_val_t)

# %%
for i in range(6):
    plt.plot(abs_pred_t[i,:,0], abs_pred_t[i,:,1],'k')

plt.plot(gt_val_t[0,obs_len:,0], gt_val_t[0,obs_len:,1],'g')
plt.plot(gt_val_t[0,:obs_len,0], gt_val_t[0,:obs_len,1],'orange')
plt.show()

#%%
import map_api as map
# %%
#550,730,1070
# plt.figure(dpi=200)
map_fpath="./maps/lanelet2/Town03.osm"
roads = map.load_lane_segments_from_xml(map_fpath)
for road_id in roads.keys():
    plt.plot(roads[road_id].l_bound[:,0], roads[road_id].l_bound[:,1], color='0.7')#, marker='o', markerfacecolor='blue', markersize=3)
    plt.plot(roads[road_id].r_bound[:,0], roads[road_id].r_bound[:,1], color='0.7')#, marker='o', markerfacecolor='red', markersize=3)
for i in range(6):
    plt.plot(abs_pred_t[i,:,0], -abs_pred_t[i,:,1],'k')

plt.plot(gt_val_t[0,obs_len:,0], -gt_val_t[0,obs_len:,1],label="5s_gt_fut")
plt.plot(gt_val_t[0,:obs_len,0], -gt_val_t[0,:obs_len,1],label="5s_gt_hist")
plt.xlim((-50, 50))
plt.ylim((120, 200))
plt.legend()
plt.show()
# plt.savefig("result_sub_3.jpg")

# %%
