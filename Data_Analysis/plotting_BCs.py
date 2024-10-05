import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def plot_Kernels(dataframe, clustering_name,  nb_clusters = None, savefig = False, path_ROIs = None, path_means = None):
    size_kernel = len(dataframe['uv_center'].iloc[0])
    assert size_kernel == 1250
    labels = ['UV-c', 'UV-s', 'G-c', 'G-s']
    colors = ['purple', 'magenta', 'darkgreen', 'lime']
    cluster_IDs = dataframe[clustering_name].to_numpy()
    assert np.amin(cluster_IDs) == 0
    if nb_clusters is None:
        nb_clusters = np.unique(cluster_IDs).shape[0]
    orderROIs = np.argsort(cluster_IDs)
    cluster_sizes = np.zeros(nb_clusters)
    for index in range(nb_clusters):
        cluster_sizes[index] = np.where(cluster_IDs == index)[0].shape[0]    
    assert cluster_sizes.sum() == len(dataframe)
    data = np.concatenate((np.vstack(dataframe['uv_center'].to_numpy()),
                           np.vstack(dataframe['uv_surround'].to_numpy()),
                           np.vstack(dataframe['green_center'].to_numpy()),
                           np.vstack(dataframe['green_surround'].to_numpy())), axis = 1)
    
    plt.figure(figsize=(8,8))
    plt.imshow(data[orderROIs,:], aspect = 'auto', cmap = 'binary_r', interpolation = 'None')
    plt.xticks(np.array([(size_kernel/2)+i*size_kernel for i in range(4)]) - 0.5, labels)
    plt.yticks(np.array([cluster_sizes[i]/2 + cluster_sizes[:i].sum() for i in range(nb_clusters)]) - 0.5,
               ['C$_{' +  str(i) + "}$" for i in range(nb_clusters)])
    plt.tick_params(length = 0)
    for index in range(1, 4):
        plt.axvline((index*size_kernel)-0.5, color = 'white', linewidth = 0.75)
    current_line_location = -0.5
    for cluster_id in range(nb_clusters-1):
        current_line_location = current_line_location + cluster_sizes[cluster_id]
        plt.axhline(current_line_location, color = 'white', linewidth = 0.75)
    if savefig:
        plt.savefig(path_ROIs, dpi = 600, transparent=True, bbox_inches='tight')
    plt.show()
    
    cluster_means = np.zeros((nb_clusters,size_kernel*4))
    for current_cluster_ID in range(nb_clusters): # Calculate cluster means
        cluster_mask = np.where(cluster_IDs == current_cluster_ID)[0]
        current_data = data[cluster_mask,:]
        cluster_means[current_cluster_ID,:]  = np.mean(current_data, axis = 0)
        
    fig, ax = plt.subplots(nb_clusters, 4, sharex='all', sharey='all', figsize=(4,8))
    for current_cluster_ID in range(nb_clusters):
        for current_kernel_ID in range(4):
            start_kernel = current_kernel_ID*size_kernel
            my_ax = ax[current_cluster_ID, current_kernel_ID]
            my_ax.axis('off')
            my_ax.axhline(0, color = 'black', linestyle = 'dashed', linewidth = 0.75)
            my_ax.axvline(size_kernel*3/4, color = 'black', linestyle = 'dashed', linewidth = 0.75) # Response
            my_ax.plot(cluster_means[current_cluster_ID,start_kernel:start_kernel+size_kernel],
                       color = colors[current_kernel_ID], linewidth = 1.5)
            if current_cluster_ID == 0:
                my_ax.set_title(labels[current_kernel_ID])
    if savefig:
        plt.savefig(path_means, dpi = 600, transparent=True, bbox_inches='tight')
    plt.show()