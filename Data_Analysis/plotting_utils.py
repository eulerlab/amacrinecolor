import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd

def bayes_factor(gmm_results):
    
    average_bic = gmm_results.groupby(['Covariance type', 'Nb components'], as_index=False).BIC.mean()
    columns = ['Seed', 'Covariance type', 'Nb components', 'Predictions', 'Bayes Factor']
    final_results = pd.DataFrame(columns = columns)

    for covariance in ['diag', 'full', 'tied', 'spherical']:
        bayes = []
        for i in range(1, np.amax(gmm_results['Nb components'].to_numpy())):

            bic1 = average_bic['BIC'][(average_bic['Covariance type'] == covariance) & 
                                      (average_bic['Nb components'] == i)].to_numpy()[0]
            bic2 = average_bic['BIC'][(average_bic['Covariance type'] == covariance) & 
                                      (average_bic['Nb components'] == i+1)].to_numpy()[0]
            bayes.append(2*(bic1-bic2))

        try:
            stop = np.amin(np.where(np.array(bayes) <= 6)[0])+1
        except: 
            stop = np.amax(gmm_results['Nb components'].to_numpy()) # Bayes factor never drops below 6


        small_df = gmm_results[(gmm_results['Covariance type'] == covariance) & 
                               (gmm_results['Nb components'] == stop)]

        predictions = small_df['Predictions'].loc[small_df['BIC'].idxmin()]
        seed = small_df['Seed'].loc[small_df['BIC'].idxmin()]

        current_data = [seed, covariance, stop, predictions, bayes]
        final_results = pd.concat([final_results, pd.DataFrame([current_data], columns=columns)], 
                                  ignore_index=True)
         
    return final_results

def plot_BIC(dataframe):
    
    best_seed = dataframe.loc[dataframe.groupby(['Covariance type', 'Nb components'])['BIC'].idxmin()]
    best_overall = dataframe.loc[dataframe.groupby(['Covariance type'])['BIC'].idxmin()]
    
    sns.lineplot(data = dataframe, x = "Nb components", y = "BIC", hue = 'Covariance type',
                 palette =['#fdbb84', '#ef6548', '#d7301f', '#7f0000'])
    plt.show()
    
    fig, axes = plt.subplots(2, 2, figsize=(13,8))
    sns.lineplot(data=best_seed[best_seed['Covariance type'] == 'full'],
                 x="Nb components", y="BIC", hue = 'Covariance type', palette = ['black'], ax=axes[0,0])
    sns.lineplot(data=dataframe[dataframe['Covariance type'] == 'full'], 
                 x="Nb components", y="BIC", hue = 'Covariance type', palette = ['#fdbb84'], ax=axes[0,0])
    
    sns.lineplot(data=best_seed[best_seed['Covariance type'] == 'tied'],
                 x="Nb components", y="BIC", hue = 'Covariance type', palette = ['black'], ax=axes[0,1])
    sns.lineplot(data=dataframe[dataframe['Covariance type'] == 'tied'], 
                 x="Nb components", y="BIC", hue = 'Covariance type', palette = ['#ef6548'], ax=axes[0,1])
    
    sns.lineplot(data=best_seed[best_seed['Covariance type'] == 'diag'],
                 x="Nb components", y="BIC", hue = 'Covariance type', palette = ['black'], ax=axes[1,0])
    sns.lineplot(data=dataframe[dataframe['Covariance type'] == 'diag'], 
                 x="Nb components", y="BIC", hue = 'Covariance type', palette = ['#d7301f'], ax=axes[1,0])
    
    sns.lineplot(data=best_seed[best_seed['Covariance type'] == 'spherical'],
                 x="Nb components", y="BIC", hue = 'Covariance type', palette = ['black'], ax=axes[1,1])
    sns.lineplot(data=dataframe[dataframe['Covariance type'] == 'spherical'], 
                 x="Nb components", y="BIC", hue = 'Covariance type', palette = ['#7f0000'], ax=axes[1,1])
    plt.show()
    
    return best_overall
    
def plot_Chirp(dataframe, clustering_name, nb_clusters = None, savefig = False, path_ROIs = None, path_means = None):
    size_chirp = len(dataframe['global_chirp'].iloc[0])
    assert size_chirp == len(dataframe['local_chirp'].iloc[0])
    labels = ['global_chirp', 'local_chirp']
    colors = ['#fdbb84', '#e34a33']
    cluster_IDs = dataframe[clustering_name].to_numpy()
    assert np.amin(cluster_IDs) == 0
    if nb_clusters is None:
        nb_clusters = np.unique(cluster_IDs).shape[0]
    orderROIs = np.argsort(cluster_IDs)
    cluster_sizes = np.zeros(nb_clusters)
    for index in range(nb_clusters):
        cluster_sizes[index] = np.where(cluster_IDs == index)[0].shape[0]    
    assert cluster_sizes.sum() == len(dataframe)
    data = np.concatenate((np.vstack(dataframe['global_chirp'].to_numpy()),
                           np.vstack(dataframe['local_chirp'].to_numpy())), axis = 1)
    
    plt.figure(figsize=(8,8))
    plt.imshow(data[orderROIs,:], aspect = 'auto', cmap = 'binary_r')
    plt.xticks(np.array([(size_chirp/2)+i*size_chirp for i in range(2)]) - 0.5, labels)
    plt.yticks(np.array([cluster_sizes[i]/2 + cluster_sizes[:i].sum() for i in range(nb_clusters)]) - 0.5,
               ['C$_{' +  str(i) + "}$" for i in range(nb_clusters)])
    plt.tick_params(length = 0)
    plt.axvline(size_chirp-0.5, color = 'white', linewidth = 0.75)
    current_line_location = -0.5
    for cluster_id in range(nb_clusters-1):
        current_line_location = current_line_location + cluster_sizes[cluster_id]
        plt.axhline(current_line_location, color = 'white', linewidth = 0.75)
    if savefig:
        plt.savefig(path_ROIs, dpi = 600, transparent=True, bbox_inches='tight')
    plt.show()
    
    cluster_means = np.zeros((nb_clusters,size_chirp*2))
    for current_cluster_ID in range(nb_clusters): # Calculate cluster means
        cluster_mask = np.where(cluster_IDs == current_cluster_ID)[0]
        current_data = data[cluster_mask,:]
        cluster_means[current_cluster_ID,:]  = np.mean(current_data, axis = 0)
        
    fig, ax = plt.subplots(nb_clusters, 2, sharex='all', sharey='all', figsize=(8,8))
    for current_cluster_ID in range(nb_clusters):
        for current_kernel_ID in range(2):
            start_kernel = current_kernel_ID*size_chirp
            my_ax = ax[current_cluster_ID, current_kernel_ID]
            my_ax.axis('off')
            my_ax.axhline(0, color = 'black', linestyle = 'dashed', linewidth = 0.75)
            my_ax.plot(cluster_means[current_cluster_ID,start_kernel:start_kernel+size_chirp],
                       color = colors[current_kernel_ID], linewidth = 1.5)
            if current_cluster_ID == 0:
                my_ax.set_title(labels[current_kernel_ID])
    if savefig:
        plt.savefig(path_means, dpi = 600, transparent=True, bbox_inches='tight')
    plt.show()