import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import os
import numpy as np
from models import FullBCModel
from utils import torch_correlation
from utils_training import *

def train(
        model,
        # Shared BCs and ACs
        stimulus_local,
        responses_local_bcs,
        responses_local_acs,
        stimulus_global,
        responses_global_bcs,
        responses_global_acs,
        # BCs
        stimulus_uv_bcs,
        responses_uv_bcs,
        stimulus_green_bcs,
        responses_green_bcs,
        # ACs
        stimulus_uv_acs,
        responses_uv_acs,
        stimulus_green_acs,
        responses_green_acs,
        penalty_matrix,
        penalty_matrix_acs,
        log_dir='.',  # where to save the model
        time_reg_weight=1e1,  # 3e-2
        sparsity_reg_weight=1e-3,  # for BCN model (before: 1e-3)
        scaling_mean_weight=1e0,  # to keep release means similar
        scaling_std_weight=1e0,  # to keep release std.dev. similar
        scaling_sc_weight = 1.0,
        lr=1e-1,  # learning rate
        noise_scale=0,
        max_steps=500,
        decrease_lr_after=3,  # if loss went up x times, lower lr
        stop_after=5,  # if lr was lowered x times, stop training
):

    params_before = {}
    for n in model.named_parameters():
        if n[1].requires_grad:
            if 'log_' in n[0]:
                param = np.exp(n[1].cpu().detach().numpy())
                name = n[0].replace('log_', '')
            else:
                param = n[1].cpu().detach().numpy()
                name = n[0]
            params_before[name] = param
    np.save(log_dir + '/params_before.npy', params_before)

    device = torch.device("cpu")
    model = model.to(device)
    penalty_matrix = torch.tensor(penalty_matrix, requires_grad = False).to(device)
    penalty_matrix_acs = torch.tensor(penalty_matrix_acs, requires_grad = False).to(device)

    # Shared BCs and ACs
    x1 = torch.tensor(stimulus_local.astype(np.float32)).to(device)
    y1_bcs = torch.tensor(responses_local_bcs.astype(np.float32)).to(device)
    y1_acs = torch.tensor(responses_local_acs.astype(np.float32)).to(device)

    x2 = torch.tensor(stimulus_global.astype(np.float32)).to(device)
    y2_bcs = torch.tensor(responses_global_bcs.astype(np.float32)).to(device)
    y2_acs = torch.tensor(responses_global_acs.astype(np.float32)).to(device)

    # BCs
    x3 = torch.tensor(stimulus_uv_bcs.astype(np.float32)).to(device)
    y3_bcs = torch.tensor(responses_uv_bcs.astype(np.float32)).to(device)
    
    x4 = torch.tensor(stimulus_green_bcs.astype(np.float32)).to(device)
    y4_bcs = torch.tensor(responses_green_bcs.astype(np.float32)).to(device)
    
    # ACs
    x5 = torch.tensor(stimulus_uv_acs[0].astype(np.float32)).to(device)
    y5_acs = torch.tensor(responses_uv_acs[0].astype(np.float32)).to(device)
    
    x6 = torch.tensor(stimulus_green_acs[0].astype(np.float32)).to(device)
    y6_acs = torch.tensor(responses_green_acs[0].astype(np.float32)).to(device)
    
    x7 = torch.tensor(stimulus_uv_acs[1].astype(np.float32)).to(device)
    y7_acs = torch.tensor(responses_uv_acs[1].astype(np.float32)).to(device)
    
    x8 = torch.tensor(stimulus_green_acs[1].astype(np.float32)).to(device)
    y8_acs = torch.tensor(responses_green_acs[1].astype(np.float32)).to(device)
    
    start_bcs = 72
    stop_bcs = 96
    uv_kernels_bcs = torch_kernels_bcs(y3_bcs, x3[:,0:3].T)
    green_kernels_bcs = torch_kernels_bcs(y4_bcs, x4[:,3:].T)
    sc_bcs = torch_sc(uv_kernels_bcs, green_kernels_bcs, start_bcs, stop_bcs, 3)

    start_acs = 51
    stop_acs = 96
    uv_kernels_acs = torch_kernels_acs(y5_acs, x5[:,0:3].T, y7_acs, x7[:,0:3].T)
    green_kernels_acs = torch_kernels_acs(y6_acs, x6[:,3:].T, y8_acs, x8[:,3:].T)
    sc_acs = torch_sc(uv_kernels_acs, green_kernels_acs, start_acs, stop_acs, 3)

    criterion = torch_correlation
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    corr_local_big = []
    corr_global_big = []
    corr_color_bcs_big = []
    corr_color_acs_big = []
    sc_bcs_big = []
    sc_acs_big = []
    total_loss_big = []
    lr_big = []
    lowest_loss = np.inf
    not_improved = 0
    stop_count = 0

    # Training Loop
    for i in range(max_steps + 1):
        if not (i % 100):
            print(i)

        optimizer.zero_grad()
        
        # Shared BCs and ACs
        y1_hat_bcs, lnr_state1, y1_hat_acs = model.forward(x1) # Local
        y2_hat_bcs, lnr_state2, y2_hat_acs = model.forward(x2) # Global
        
        # BCs
        y3_hat_bcs, lnr_state3, _ = model.forward(x3) # UV
        y4_hat_bcs, lnr_state4, _ = model.forward(x4) # Green
        
        # ACs
        _, lnr_state5, y5_hat_acs = model.forward(x5) # UV
        _, lnr_state6, y6_hat_acs = model.forward(x6) # Green
        _, lnr_state7, y7_hat_acs = model.forward(x7) # UV
        _, lnr_state8, y8_hat_acs = model.forward(x8) # Green

        # Correlation
        loss_corr_local = 1 - criterion(
            torch.cat([y1_hat_bcs, y1_hat_acs], axis = 0), torch.cat([y1_bcs, y1_acs], axis = 0))
        loss_corr_global = 1 - criterion(
            torch.cat([y2_hat_bcs, y2_hat_acs], axis = 0), torch.cat([y2_bcs, y2_acs], axis = 0))
        
        loss_color_bcs = 1 - criterion(
            torch.cat([y3_hat_bcs, y4_hat_bcs], axis = 1), torch.cat([y3_bcs, y4_bcs], axis = 1))
        loss_color_acs = 1 - criterion(
            torch.cat([y5_hat_acs, y6_hat_acs, y7_hat_acs, y8_hat_acs], axis = 1), 
            torch.cat([y5_acs, y6_acs, y7_acs, y8_acs], axis = 1))
        
        uv_kernels_bcs_hat = torch_kernels_bcs(y3_hat_bcs, x3[:,0:3].T)
        green_kernels_bcs_hat = torch_kernels_bcs(y4_hat_bcs, x4[:,3:].T)
        sc_bcs_hat = torch_sc(uv_kernels_bcs_hat, green_kernels_bcs_hat, start_bcs, stop_bcs, 3)

        uv_kernels_acs_hat = torch_kernels_acs(y5_hat_acs, x5[:,0:3].T, y7_hat_acs, x7[:,0:3].T)
        green_kernels_acs_hat = torch_kernels_acs(y6_hat_acs, x6[:,3:].T, y8_hat_acs, x8[:,3:].T)
        sc_acs_hat = torch_sc(uv_kernels_acs_hat, green_kernels_acs_hat, start_acs, stop_acs, 3)
        
        loss_sc_bcs = scaling_sc_weight * torch.mean((sc_bcs - sc_bcs_hat)**2)
        loss_sc_acs = scaling_sc_weight * torch.mean((sc_acs - sc_acs_hat)**2)

        # regularize std.dev. of log_speeds
        reg_speed = time_reg_weight * torch.std(
            torch.exp(model.log_kernel_speed))
        # L1 penalty on connectivity
        reg_sparsity = sparsity_reg_weight * (
            torch.sum(torch.exp(model.log_acl_bc_weight)*penalty_matrix.T) +
            torch.sum(torch.exp(
                model.glycinergic_amacrine_cells.log_bc_ac_weight)*penalty_matrix) +
            torch.sum(torch.exp(model.log_pr_bc_weight)) +
            torch.sum(torch.exp(model.log_acl_acl_weight)*penalty_matrix_acs)
                )
        # scaling penalty
        releases1 = torch.cat([
            lnr_state1['track_release'], 
            lnr_state2['track_release'],
            torch.stack(lnr_state1['track_acl_output']), 
            torch.stack(lnr_state2['track_acl_output'])], dim=1)
        releases2 = torch.cat([
            torch.cat([lnr_state3['track_release'], 
                       lnr_state4['track_release']], dim = 0),
            torch.cat([torch.stack(lnr_state3['track_acl_output']),
                       torch.stack(lnr_state4['track_acl_output'])], dim = 0)], dim=1)
        releases3 = torch.cat([
            torch.cat([lnr_state5['track_release'], 
                       lnr_state6['track_release'],
                       lnr_state7['track_release'],
                       lnr_state8['track_release']], dim = 0),
            torch.cat([torch.stack(lnr_state5['track_acl_output']),
                       torch.stack(lnr_state6['track_acl_output']),
                       torch.stack(lnr_state7['track_acl_output']),
                       torch.stack(lnr_state8['track_acl_output'])], dim = 0)], dim=1)
        means = torch.cat([torch.mean(releases1, dim=0), torch.mean(releases2, dim=0), torch.mean(releases3, dim=0)], dim=0)
        stds = torch.cat([torch.std(releases1, dim=0), torch.std(releases2, dim=0), torch.std(releases3, dim=0)], dim=0)
        scaling_mean_penalty = scaling_mean_weight * torch.std(means)
        scaling_std_penalty = scaling_std_weight * torch.std(stds)

        # final loss
        loss = loss_corr_local + loss_corr_global + loss_color_bcs + loss_color_acs + \
            loss_sc_bcs + loss_sc_acs + reg_speed + reg_sparsity + scaling_mean_penalty + scaling_std_penalty
        corr_local_big.append(loss_corr_local.item())
        corr_global_big.append(loss_corr_global.item())
        corr_color_bcs_big.append(loss_color_bcs.item())
        corr_color_acs_big.append(loss_color_acs.item())
        sc_bcs_big.append(loss_sc_bcs.item())
        sc_acs_big.append(loss_sc_acs.item())
        total_loss_big.append(loss.item())
        lr_big.append(lr)

        if torch.isnan(loss):
            print('cancel because of nan')
            return total_loss_big

        # early stopping
        if loss.item() > lowest_loss:
            not_improved += 1
            if not_improved >= decrease_lr_after:
                # go back to last best
                model.load_state_dict(
                    torch.load(os.path.join(log_dir, 'model.pth')))
                stop_count += 1
                if stop_count >= stop_after:
                    break
                lr *= .5
                for g in optimizer.param_groups:  # decrease lr
                    g['lr'] *= .5
                not_improved = 0
            else:
                # make step if it continues at this lr
                loss.backward()
                optimizer.step()
        else:
            torch.save(
                model.state_dict(), os.path.join(log_dir, 'model.pth'))
            lowest_loss = loss.item()
            # make step after saving best
            loss.backward()
            optimizer.step()

        for p in model.parameters():
            s = lr * noise_scale
            if p.requires_grad:
                if p.shape[0] > 1:
                    p.data += torch.randn_like(p.data) * s * torch.std(p.data)
                else:
                    p.data += torch.randn_like(p.data) * s

    params_after = {}
    for n in model.named_parameters():
        if n[1].requires_grad:
            if 'log_' in n[0]:
                param = np.exp(n[1].cpu().detach().numpy())
                name = n[0].replace('log_', '')
            else:
                param = n[1].cpu().detach().numpy()
                name = n[0]
            params_after[name] = param
    
    np.save(log_dir + '/params_after.npy', params_after)
    np.save(log_dir + '/total_loss.npy', total_loss_big)
    np.save(log_dir + '/corr_loss_local.npy', corr_local_big)
    np.save(log_dir + '/corr_loss_global.npy', corr_global_big)
    np.save(log_dir + '/corr_loss_color_bcs.npy', corr_color_bcs_big)
    np.save(log_dir + '/corr_loss_color_acs.npy', corr_color_acs_big)
    np.save(log_dir + '/sc_loss_bcs.npy', sc_bcs_big)
    np.save(log_dir + '/sc_loss_acs.npy', sc_acs_big)
    np.save(log_dir + '/lr.npy', lr_big)

    return total_loss_big