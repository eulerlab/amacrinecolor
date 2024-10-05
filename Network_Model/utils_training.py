import numpy as np
import pandas as pd
from scipy import stats
import torch

def torch_kernels_bcs(traces, stimulus):
    line_duration = 1/64
    kernel_length_s = 2.0
    nb_conditions = stimulus.shape[0]
    kernel_length_line = np.int(np.floor(kernel_length_s / line_duration))
    offset_after = np.int(np.floor(kernel_length_line * .25))
    offset_before = kernel_length_line - offset_after

    full_traces = traces[:, offset_before:-offset_after]
    full_matrix = torch.zeros((kernel_length_line, full_traces.shape[1], stimulus.shape[0]), dtype=torch.float32)
    trace_length = full_traces.shape[1]
    for i in range(kernel_length_line):
        full_matrix[i, :, :] = stimulus[:, i:trace_length + i].T
    # Z scoring traces and stimulus matrix
    for i in range(nb_conditions):
        full_matrix[:, :, i] = (full_matrix[:, :, i].T - torch.mean(full_matrix[:, :, i], axis=1)).T
        full_matrix[:, :, i] = (full_matrix[:, :, i].T / torch.std(full_matrix[:, :, i], axis=1)).T
    full_traces = (full_traces.T - torch.mean(full_traces, axis=1)).T
    full_traces = (full_traces.T / torch.std(full_traces, axis=1)).T
    kernels_all_rois = torch.matmul(full_traces, full_matrix)
    kernels_all_rois /=full_traces.shape[1]
    return kernels_all_rois

def torch_kernels_acs(first_traces, first_stim, second_traces, second_stim):
    line_duration = 1/64
    kernel_length_s = 2.0
    nb_conditions = first_stim.shape[0]
    kernel_length_line = np.int(np.floor(kernel_length_s / line_duration))
    offset_after = np.int(np.floor(kernel_length_line * .25))
    offset_before = kernel_length_line - offset_after
    
    first_traces_cropped = first_traces[:, offset_before:-offset_after]
    second_traces_cropped = second_traces[:, offset_before:-offset_after]
    design_matrix1 = torch.zeros((kernel_length_line, first_traces_cropped.shape[1], first_stim.shape[0]), dtype=torch.float32)
    trace_length = first_traces_cropped.shape[1]
    for i in range(kernel_length_line):
        design_matrix1[i, :, :] = first_stim[:, i:trace_length + i].T
    design_matrix2 = torch.zeros((kernel_length_line, second_traces_cropped.shape[1], second_stim.shape[0]),
                                 dtype=torch.float32)
    trace_length = second_traces_cropped.shape[1]
    for i in range(kernel_length_line):
        design_matrix2[i, :, :] = second_stim[:, i:trace_length + i].T
    # concatenate the first and second parts of the traces and stimulus matrices
    full_matrix = torch.cat([design_matrix1, design_matrix2], axis=1)
    full_traces = torch.cat([first_traces_cropped, second_traces_cropped], axis=1)
    for i in range(nb_conditions):
        full_matrix[:, :, i] = (full_matrix[:, :, i].T - torch.mean(full_matrix[:, :, i], axis=1)).T
        full_matrix[:, :, i] = (full_matrix[:, :, i].T / torch.std(full_matrix[:, :, i], axis=1)).T
    full_traces = (full_traces.T - torch.mean(full_traces, axis=1)).T
    full_traces = (full_traces.T / torch.std(full_traces, axis=1)).T
    kernels_all_rois = torch.matmul(full_traces, full_matrix)
    kernels_all_rois /=full_traces.shape[1]
    return kernels_all_rois

def torch_sc(uv_kernels, green_kernels, start, stop, nb_conditions):
    size_kernel = uv_kernels.shape[0]
    nb_clusters = uv_kernels.shape[1]
    sc = torch.zeros((nb_clusters,nb_conditions), dtype=torch.float64)
    
    for i in range(nb_conditions):
        
        uv = uv_kernels[:,:,i].T
        green = green_kernels[:,:,i].T
        assert uv.shape[1] == size_kernel
        assert green.shape[1] == size_kernel
        
        min_uv = torch.min(uv[:,start:stop], axis = 1)[0]
        max_uv = torch.max(uv[:,start:stop], axis = 1)[0]
        min_green = torch.min(green[:,start:stop], axis = 1)[0]
        max_green = torch.max(green[:,start:stop], axis = 1)[0]
        amp_uv = max_uv - min_uv
        amp_green = max_green - min_green
        sc[:,i] = (amp_green - amp_uv)/(amp_green + amp_uv)
     
    return sc