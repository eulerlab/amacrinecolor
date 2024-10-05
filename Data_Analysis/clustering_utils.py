from sklearn.mixture import GaussianMixture
from sklearn.model_selection import KFold
import numpy as np
import pandas as pd

def fit_gmm(data, max_nb_components, nb_replicates, max_iterations):
    
    columns = ['Seed', 'Covariance type', 'Nb components', 'Max iterations', 'BIC', 'Predictions']
    gmm_results = pd.DataFrame(columns = columns)

    for covariance_type in ['full', 'tied', 'diag', 'spherical']:
        for nb_of_components in np.arange(1,1+max_nb_components):
            for replicate in np.arange(nb_replicates):

                gmm = GaussianMixture(n_components=nb_of_components,
                                      covariance_type=covariance_type,
                                      max_iter=max_iterations,                                     
                                      random_state=replicate)
                labels = gmm.fit_predict(data)
                BIC = gmm.bic(data)

                current_data = {'Seed': replicate,
                                'Covariance type': covariance_type,
                                'Nb components': nb_of_components,
                                'Max iterations': max_iterations,
                                'BIC': BIC,
                                'Predictions': labels}
                
                gmm_results = pd.concat([gmm_results, pd.DataFrame([current_data], columns=columns)],
                                        ignore_index=True)
    
    return gmm_results

def permute_clusterIDs(dataframe, clustering_name):
    
    cluster_IDs = dataframe[clustering_name].to_numpy()
    assert np.amin(cluster_IDs) == 0
    nb_clusters = np.unique(cluster_IDs).shape[0]
    IPL_depth = dataframe['ipl_depth'].to_numpy()
    
    cluster_locations = np.zeros(nb_clusters)
    for current_cluster in range(nb_clusters):
        clusterMask = np.where(cluster_IDs == current_cluster)[0]
        cluster_locations[current_cluster] = np.median(IPL_depth[clusterMask])
    
    new_cluster_IDs = 100*np.ones(len(dataframe))
    for new_ID, old_ID in enumerate(np.argsort(cluster_locations)):
        new_cluster_IDs[np.where(cluster_IDs == old_ID)[0]] = int(new_ID)
        
    return new_cluster_IDs

def cluster_average(dataframe, clustering_name, label_name):
    
    cluster_IDs = dataframe[clustering_name].to_numpy()
    assert np.amin(cluster_IDs) == 0
    nb_clusters = np.unique(cluster_IDs).shape[0]
    data = np.vstack(dataframe[label_name].to_numpy())
    cluster_means = np.zeros((nb_clusters,data.shape[-1]))
    
    for current_cluster in range(nb_clusters): # Calculate cluster means
        clusterMask = np.where(cluster_IDs == current_cluster)[0]
        cluster_means[current_cluster,:]  = np.mean(data[clusterMask,:], axis = 0)
    
    return cluster_means

def cv_gmm(data, nb_splits, max_nb_components, nb_replicates, max_iterations):

    columns = ['Split', 'Seed', 'Covariance type', 'Nb components', 'BIC - Train', 'LL - Test']
    gmm_results = pd.DataFrame(columns = columns)
    
    kf = KFold(n_splits = nb_splits)
    split_index = 0
    for train_index, test_index in kf.split(data):
        X_train = data[train_index,:]
        X_test = data[test_index,:]
        
        for covariance_type in ['full', 'tied', 'diag', 'spherical']:
            for nb_of_components in np.arange(1,1+max_nb_components):
                for replicate in np.arange(nb_replicates):

                    gmm = GaussianMixture(n_components=nb_of_components,
                                          covariance_type=covariance_type,
                                          max_iter=max_iterations,                                     
                                          random_state=replicate)

                    gmm.fit(X_train)
                    BIC = gmm.bic(X_train)
                    score = gmm.score(X_test)

                    current_data = {'Split': split_index,
                                    'Seed': replicate,
                                    'Covariance type': covariance_type,
                                    'Nb components': nb_of_components,
                                    'BIC - Train': BIC,
                                    'LL - Test': score}

                    gmm_results = pd.concat([gmm_results, pd.DataFrame([current_data], columns=columns)],
                                            ignore_index=True)
        
        split_index += 1

    return gmm_results