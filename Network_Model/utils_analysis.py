import numpy as np
from matplotlib.collections import PatchCollection
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import pandas as pd
from scipy import stats

def estimate_kernels_bcs(traces, stimulus):
    line_duration = 1/64
    kernel_length_s = 2.0
    nb_conditions = stimulus.shape[0]
    kernel_length_line = np.int(np.floor(kernel_length_s / line_duration))
    offset_after = np.int(np.floor(kernel_length_line * .25))
    offset_before = kernel_length_line - offset_after

    full_traces = traces[:, offset_before:-offset_after]
    full_matrix = np.zeros((kernel_length_line, full_traces.shape[1], stimulus.shape[0]))
    trace_length = full_traces.shape[1]
    for i in range(kernel_length_line):
        full_matrix[i, :, :] = stimulus[:, i:trace_length + i].T
    # Z scoring traces and stimulus matrix
    for i in range(nb_conditions):
        full_matrix[:, :, i] = (full_matrix[:, :, i].T - np.mean(full_matrix[:, :, i], axis=1)).T
        full_matrix[:, :, i] = (full_matrix[:, :, i].T / np.std(full_matrix[:, :, i], axis=1)).T
    full_traces = (full_traces.T - np.mean(full_traces, axis=1)).T
    full_traces = (full_traces.T / np.std(full_traces, axis=1)).T
    kernels_all_rois = np.matmul(full_traces, full_matrix)
    kernels_all_rois /=full_traces.shape[1]
    return kernels_all_rois

def estimate_kernels_acs(first_traces, first_stim, second_traces, second_stim):
    line_duration = 1/64
    kernel_length_s = 2.0
    nb_conditions = first_stim.shape[0]
    kernel_length_line = np.int(np.floor(kernel_length_s / line_duration))
    offset_after = np.int(np.floor(kernel_length_line * .25))
    offset_before = kernel_length_line - offset_after
    
    first_traces_cropped = first_traces[:, offset_before:-offset_after]
    second_traces_cropped = second_traces[:, offset_before:-offset_after]
    design_matrix1 = np.zeros((kernel_length_line, first_traces_cropped.shape[1], first_stim.shape[0]))
    trace_length = first_traces_cropped.shape[1]
    for i in range(kernel_length_line):
        design_matrix1[i, :, :] = first_stim[:, i:trace_length + i].T
    design_matrix2 = np.zeros((kernel_length_line, second_traces_cropped.shape[1], second_stim.shape[0]))
    trace_length = second_traces_cropped.shape[1]
    for i in range(kernel_length_line):
        design_matrix2[i, :, :] = second_stim[:, i:trace_length + i].T
    # concatenate the first and second parts of the traces and stimulus matrices
    full_matrix = np.concatenate((design_matrix1, design_matrix2), axis=1)
    full_traces = np.concatenate((first_traces_cropped, second_traces_cropped), axis=1)
    for i in range(nb_conditions):
        full_matrix[:, :, i] = (full_matrix[:, :, i].T - np.mean(full_matrix[:, :, i], axis=1)).T
        full_matrix[:, :, i] = (full_matrix[:, :, i].T / np.std(full_matrix[:, :, i], axis=1)).T
    full_traces = (full_traces.T - np.mean(full_traces, axis=1)).T
    full_traces = (full_traces.T / np.std(full_traces, axis=1)).T
    kernels_all_rois = np.matmul(full_traces, full_matrix)
    kernels_all_rois /=full_traces.shape[1]
    return kernels_all_rois
    
def estimate_sc(uv_kernels, uv_kernels_hat, green_kernels, green_kernels_hat, start, stop):
    size_kernel = uv_kernels.shape[0]
    nb_clusters = uv_kernels.shape[1]
    nb_conditions = uv_kernels.shape[2]
    sc = np.zeros((nb_clusters,nb_conditions))
    sc_hat = np.zeros((nb_clusters,nb_conditions))
    
    for i in range(nb_conditions):
        
        uv = uv_kernels[:,:,i].T
        green = green_kernels[:,:,i].T
        uv_hat = uv_kernels_hat[:,:,i].T
        green_hat = green_kernels_hat[:,:,i].T
        
        assert uv.shape[1] == size_kernel
        assert uv_hat.shape[1] == size_kernel
        assert green.shape[1] == size_kernel
        assert green_hat.shape[1] == size_kernel
        
        min_uv = np.amin(uv[:,start:stop], axis = 1)
        max_uv = np.amax(uv[:,start:stop], axis = 1)
        min_uv_hat = np.amin(uv_hat[:,start:stop], axis = 1)
        max_uv_hat = np.amax(uv_hat[:,start:stop], axis = 1)
        amp_uv = max_uv - min_uv
        amp_uv_hat = max_uv_hat - min_uv_hat

        min_green = np.amin(green[:,start:stop], axis = 1)
        max_green = np.amax(green[:,start:stop], axis = 1)
        min_green_hat = np.amin(green_hat[:,start:stop], axis = 1)
        max_green_hat = np.amax(green_hat[:,start:stop], axis = 1)
        amp_green = max_green - min_green
        amp_green_hat = max_green_hat - min_green_hat

        sc[:,i] = (amp_green - amp_uv)/(amp_green + amp_uv)
        sc_hat[:,i] = (amp_green_hat - amp_uv_hat)/(amp_green_hat + amp_uv_hat)
     
    return sc, sc_hat