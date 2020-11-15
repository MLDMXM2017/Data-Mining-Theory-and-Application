import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda
import time
import datetime
from pycuda.compiler import SourceModule
from .kernels import clinicalKernel as ck

def getSilhouettes(inData, catCols=np.array([],dtype=int), clusterLabels=np.array([], dtype=int), save_to_csv=""):
    if clusterLabels.size == 0:
        raise ValueError("Input 'clusterLabels' is empty")
    if inData.shape[0] != clusterLabels.shape[0]:
        raise ValueError("Shape mismatch: number of samples in input 'inData' does not equal that of input 'clusterLabels'")
    print("\n#### getSilhouettes ####\n")
    ### Computing aPrime[i]: within cluster similarity
    unique_clusters = np.unique(clusterLabels)
    num_unique_clusters = unique_clusters.size
    if num_unique_clusters < 2 :
        raise ValueError("Number of unique clusters in input 'clusterLabels' is less than 2")
    if catCols.size == 0:
        print("# No categorical columns given in input. Assuming all to be numerical.")
    print("# Total number of clusters: "+repr(num_unique_clusters))
    a_prime = np.zeros(inData.shape[0], dtype=np.float32, order='C')
    b_prime = np.zeros(inData.shape[0], dtype=np.float32, order='C')

    ## Comute within-cluster similarities ##
    print("## Computing within-cluster similarities ##\n")
    for clusterIter in range(num_unique_clusters):
        # print("\n## Cluster "+repr(clusterIter))
        cluster_mask = clusterLabels == unique_clusters[clusterIter]
        cluster_data = inData[cluster_mask, :]
        cluster_similarity_matrix = ck.clinicalKernel(cluster_data, catCols = catCols, return_to_CPU = True)
        num_data_cluster = cluster_data.shape[0]
        a_prime[cluster_mask] = (1/(num_data_cluster-1))*(np.sum(cluster_similarity_matrix, axis=1))

    b_prime_allclusters = np.zeros((inData.shape[0], num_unique_clusters))

    print("\n\n## Computing between-cluster similarities ##")
    for iter1 in range(num_unique_clusters):
        for iter2 in range(num_unique_clusters):
            if(iter1 != iter2): #Because we don't want to compare within cluster
                # print("\n## Cluster "+repr(iter1)+" and Cluster "+repr(iter2))
                cluster_mask_1 = (clusterLabels == unique_clusters[iter1])
                cluster_data_1 = inData[cluster_mask_1, :]
                cluster_mask_2 = (clusterLabels == unique_clusters[iter2])
                cluster_data_2 = inData[cluster_mask_2, :]
                cluster_data_double = np.concatenate((cluster_data_1, cluster_data_2), axis=0)
                similarity_double = ck.clinicalKernel(cluster_data_double, catCols=catCols, return_to_CPU = True)
                b_prime_allclusters[cluster_mask_1, iter2] = (1/cluster_data_2.shape[0])*(np.sum(similarity_double[:cluster_data_1.shape[0],:], axis=1)-((cluster_data_1.shape[0]-1)*a_prime[cluster_mask_1]))

    b_prime = np.amax(b_prime_allclusters, axis=1)
    # print("\n## Done. ##\n")
    silhouette = np.zeros_like(a_prime)
    silhouette = (a_prime - b_prime)/np.maximum(a_prime, b_prime)
    silhouette_score = (1/inData.shape[0])*np.sum(silhouette)
    print("# Silhouette score for data = "+repr(silhouette_score))
    timeNow = datetime.datetime.now()
    timeString = timeNow.strftime("%H:%M:%S %Y-%m-%d")
    headerString = "#### Silhouettes ####\n# "+timeString+"\n# Number of clusters = "+repr(num_unique_clusters)+"\n# Silhouette score = "+repr(silhouette_score)
    if not save_to_csv:
        save_to_csv = "silhouettes_"+repr(num_unique_clusters)+"clusters.csv"
    np.savetxt(save_to_csv, silhouette, delimiter=",", header=headerString, comments="")
    print("# Silhouette data saved to "+save_to_csv)
    return silhouette
