import torch
device = "cuda" if torch.cuda.is_available() else "cpu"

def loss_mm(y_pred, y_true, log_probs, anchors):
    """
    params:
    :y_true: N_T x N_B x 2, actual XY trajectory taken
    GMM mode trajectory parameters [mux_ak, muy+ak, log(std1), log(std2), theta]
    and probabilities(anchor probs) N_B x (N_M x (1 + 5*N_T))
    :y_pred: N_T x N_M x N_B X 5,
    :log_probs: N_B x N_M
    :anchors: N_M x N_T x 2
    where N_B is batch_size, N_T is target_len, N_M is num_modes
    """
    #N_T x N_B x 2
    y_true = y_true.permute(1,0,2)

    batch_size = y_true.shape[0]
    #N_B x N_M x N_T X 5
    trajectories = y_pred.permute(2,1,0,3)
    anchor_probs = log_probs

    #find the nearest anchor mode (k_cl) (anchors-y_true)^2
    #N_M x N_T x 2 - N_B x 1 x N_T x 2 = N_B x N_M x N_T x 2
    distance_to_anchors = torch.sum(torch.linalg.vector_norm(anchors.to(device)-y_true.unsqueeze(1).to(device), 
                            dim=-1), dim=-1) #N_B x N_M
    nearest_mode = distance_to_anchors.argmin(dim=-1) #N_B

    loss_cls = -log_probs[torch.arange(batch_size),nearest_mode].squeeze() #N_B

    nearest_trajs = trajectories[torch.arange(batch_size),nearest_mode,:,:] #N_B x N_T x 5
    residual_trajs = y_true.to(device) - nearest_trajs[:,:,:2].to(device)

    # dx = residual_trajs[:,:,0]
    # dy = residual_trajs[:,:,1]
    # std1 = torch.clamp(torch.abs(nearest_trajs[:,:,2]), min=0., max=5.)
    # std1 = torch.clamp(torch.abs(nearest_trajs[:,:,3]), min=0., max=5.)

    # cos_th = torch.cos(nearest_trajs[:,:,4])
    # sin_th = torch.sin(nearest_trajs[:,:,4])
    loss_reg = torch.pow(residual_trajs, exponent=2)
    loss_reg = torch.sum(loss_reg, dim=2)
    loss_reg = torch.pow(loss_reg, exponent=0.5)
    loss_reg = torch.sum(loss_reg, dim=1) #N_B

    total_loss = loss_reg + loss_cls

    return loss_reg, loss_cls













