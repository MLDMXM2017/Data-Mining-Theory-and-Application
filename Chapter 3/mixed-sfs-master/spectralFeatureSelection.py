import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda
import time
from pycuda.compiler import SourceModule
import pycuda.gpuarray as gpuarray

import datetime
from kernels import clinicalKernel as ck
import laplacian

from numpy.linalg import *

def getEigenValues(inData, maxNumClusters, catCols = np.array([], dtype=int), save_to_file="eigenValues.csv", additional_comments=""):
    ### Input:
    ### # inData: (numpy.ndarray) Array of shape (numSamples, numFeatures)
    ### # maxNumClusters: (int) Maximum number of clusters we want
    ### # catCols: (numpy.ndarray, dtype=int) 1-D array containing indices of categorical columns in inData
    ### # save_to_file: (str) Filepath to save the eigenvalues. Default: ./eigenValues.csv
    ### # additional_comments: (str) Any additional comments to be added in the header of output file. (e.g. "Data that I collected last Monday")
    ###
    ### Output:
    ### # eigVals: (numpy.ndarray) Array of shape (maxNumClusters+1, numFeatures+1). eigVals[:,i] (i>0) contains the first maxNumClusters+1 eigenvalues of the laplacian with i'th feature removed (i = 1, 2, ... numFeatures). eigVals[:,0] contains the first maxNumClusters+1 eigenvalues of the laplacian with no feature removed.
    ###
    ### Description:
    ### # Gets eigenValues of the laplacian matrix for input data inData, with catCols. This is done iteratively numFeatures+1 ( = inData.shape[1]+1) times.
    ###
    ### Note:
    ### # This function is expected to take a long time to execute, depending on the size on inData. Suggest calling this function on smaller data first, and proceed on actual data, if all is good.
    start_time = time.time()
    if np.argwhere(np.isnan(inData) is True).size !=0:
        raise ValueError("Input array inData contains NaN entries.")
        return 0
    inData = np.asanyarray(inData, dtype=np.float32, order='C')
    numSamples = inData.shape[0]
    numFeatures = inData.shape[1]

    eigVals = np.zeros((maxNumClusters+1, numFeatures+1), dtype=np.float32, order='C')
    print("\n#### Working on base data\n")
    similarity_matrix_gpu = ck.clinicalKernel(inData, catCols = catCols, return_to_CPU = False)

    laplacian_matrix = laplacian.laplacian_normalised(similarity_matrix_gpu, numRows = numSamples, simMatrix_on_CPU = False, return_to_CPU = True)

    ### Change the order of laplacian_matrix from 'C' to 'F', because cuSolver and skcuda like 'F'
    ### Send laplacian_matrix back to GPU as pycuda.gpuarray.GPUArray, with order 'F'
    laplacian_matrix = np.asanyarray(laplacian_matrix, order='F', dtype=np.float32)
#    laplacian_matrix_gpu = gpuarray.to_gpu(laplacian_matrix)

#    linalg.init()
    print("\n# Computing eigenvalues ...")
#    eigVals_current = linalg.eig(laplacian_matrix_gpu)
    eigVals_current,y = linalg.eig(laplacian_matrix)
#    eigVals_current = eigVals_current.get() #Convert GPUArray to numpy array and sort in increasing order
    eigVals_current=np.sort(eigVals_current)
#    eigVals_current.sort()
    eigVals_current = eigVals_current[:(maxNumClusters+1)]

    eigVals[:,0] = eigVals_current

    ### Main for loop that iterates over features

    for i in range(numFeatures):
        time_elapsed = time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))
        print("\n#### Feature dropped: "+repr(i+1)+" out of "+repr(numFeatures)+" (time elapsed: "+time_elapsed+")")
        catCols_reduced = np.delete(catCols, np.nonzero(catCols == i)) #If a catCols is currently being dropped
        mask = catCols_reduced > i
        catCols_reduced[mask] = catCols_reduced[mask]-1 # When i'th feature is dropped, indices > i decrease by 1
        # print("catCols: "+repr(catCols_reduced))
        inData_reduced = np.delete(inData, i, axis=1)
        similarity_matrix_gpu = ck.clinicalKernel(inData_reduced, catCols=catCols_reduced, return_to_CPU=False)
        laplacian_matrix = laplacian.laplacian_normalised(similarity_matrix_gpu, numRows = numSamples, simMatrix_on_CPU = False, return_to_CPU = True)
        ### Change the order of laplacian_matrix from 'C' to 'F', because cuSolver and skcuda like 'F'
        ### Send laplacian_matrix back to GPU as pycuda.gpuarray.GPUArray, with order 'F'
        laplacian_matrix = np.asanyarray(laplacian_matrix, order='F', dtype=np.float32)
#        laplacian_matrix_gpu = gpuarray.to_gpu(laplacian_matrix)
        print("\n# Computing eigenvalues ...")
        eigVals_current,y = linalg.eig(laplacian_matrix)
#        eigVals_current = eigVals_current.get() #Convert GPUArray to numpy array and sort in increasing order
        eigVals_current=np.sort(eigVals_current)
#        eigVals_current.sort()
        eigVals_current = eigVals_current[:(maxNumClusters+1)]
        eigVals[:,(i+1)] = eigVals_current

    timeNow = datetime.datetime.now()
    timeString = timeNow.strftime("%H:%M:%S %Y-%m-%d")
    header = "##### Eigenvalues #####\n# "+timeString+"\n# First "+repr(maxNumClusters+1)+" Eigenvalues of Laplacian.\n# Eigenvalues stored along columns. 0th column for no feature dropped, i'th column for i'th feature dropped (i > 0)\n#\n# Comment: "+str(additional_comments)
    np.savetxt(save_to_file, eigVals, header=header, comments="", delimiter=",")
    print("\n#### Iterations complete\n")
    print("\n# Eigenvalues saved to file "+save_to_file)
    return eigVals

def selectFeatures_fromEigVals(maxNumClusters, eigVals=np.array([]), eigVals_from_file=""):
    ### Input:
    ### # eigVals: (numpy.ndarray) 2-D array with shape (maxNumClusters, numFeatures). Can be optionally passed from saved file (ref. eigVals_from_file below).
    ### # eigVals_from_file: (str) Filepath to fetch eigenvales from. Empty by default.
    ###
    ### Output:
    ### # featureWeights: (numpy.ndarray) 2-D array (shape: (maxNumClusters-1, numFeatures)) with featureWeights along rows. Multiple set of feature weights (according to different numClusters, starting from 2 to maxNumClusters) are stacked along columns.
    ###
    ### Description:
    ### # Computes feature weights iteratively over the number of clusters (2 <= number of clusters <= maxNumClusters) given the eigenvalues matrix.


    #Spectral gap score of the base data, with no feature deleted. gamma0[i] corresponds to i+2 clusters
    if eigVals_from_file:
        eigVals_in = np.genfromtxt(eigVals_from_file, delimiter=",", comments="#")
        if eigVals_in.size == 0:
            raise ValueError("Input eigenvalues file "+str(eigVals_from_file)+" returned an empty array.")
    else:
        if eigVals.size == 0:
            raise ValueError("Input argument eigVals is an empty array")
        else:
            eigVals_in = eigVals
    gamma0 = np.zeros(maxNumClusters-1)
    eigVals_current = eigVals_in[:,0]
    for i in range(maxNumClusters-1):
        numClusters = i+2
        eigVec = eigVals_current[:numClusters+1]
        tau = np.sum(eigVec[1:])
        eigSubVec_1 = eigVec[1:-1]
        eigSubVec_2 = eigVec[2:]
        gamma0[i] = np.sum(np.abs((eigSubVec_1 - eigSubVec_2)/tau))

    ## Now compute spectral gap score for each numClusters < maxNumClusters, for each feature removed.
    numFeatures = eigVals_in.shape[1]-1
    featureWeights = np.zeros((maxNumClusters-1, numFeatures))
    for i in range(maxNumClusters-1):
        numClusters = i+2
        for j in range(numFeatures):
            eigVals_current = eigVals_in[:numClusters+1, j+1]
            tau = np.sum(eigVals_current[1:])
            eigSubVec_1 = eigVals_current[1:-1]
            eigSubVec_2 = eigVals_current[2:]
            featureWeights[i, j] = gamma0[i] - np.sum((np.abs(eigSubVec_1 - eigSubVec_2)/tau))
    return featureWeights


def selectFeatures(inData, maxNumClusters, catCols = np.array([], dtype=int), save_eigenvalues = "eigenValues.csv", save_featureWeights="featureWeights.csv", additional_comments = ""):
    ### Wrapper function for spectralFeatureSelection_fromEigVals and getEigenValues

    ### Input:
    ### # inData: (numpy.ndarray) Array of shape (numSamples, numFeatures)
    ### # maxNumClusters: (int) Maximum number of clusters we want
    ### # catCols: (numpy.ndarray, dtype=int) 1-D array containing indices of categorical columns in inData
    ### # save_eigenvalues: (str) Filepath to save the eigenvalues. Default: "./eigenValues.csv"
    ### # save_featureWeights: (str) Filepath to save the featureWeights. Default: "./featureWeights.csv"
    ### # additional_comments: (str) Any additional comments to be added in the header of output file. (e.g. "Data that I collected last Monday")
    ###
    ### Output:
    ### # featureWeights: (numpy.ndarray) 2-D array (shape: (maxNumClusters-1, numFeatures)) with featureWeights along rows. Multiple set of feature weights (according to different numClusters, starting from 2 to maxNumClusters) are stacked along columns.
    ###
    ### Descriptiom:
    ### # Final wrapper function that performs spectral feature selection on inData, for maximum clusters maxNumClusters, and returns featureWeights for each value of numClusters between 2 and maxNumClusters (both inclusive)
    ###
    ### Note:
    ### # This function is expected to take a long time to execute, depending on the size on inData. Suggest calling this function on smaller data first, and proceed on actual data, if all is good.

    ## getEigenValues
    eigVals = getEigenValues(inData, maxNumClusters, catCols, save_to_file=save_eigenvalues, additional_comments=additional_comments)

    ## spectralFeatureSelection_fromEigVals
    timeNow = datetime.datetime.now()
    timeString = timeNow.strftime("%H:%M:%S %Y-%m-%d")
    featureWeights = selectFeatures_fromEigVals(maxNumClusters, eigVals)
    header = "##### Feature weights #####\n# "+timeString+"\n# Feature weights stored along rows.\n# Row i (0, 1, 2..) contains feature weights assuming (i+2) clusters.\n#\n# Comment: "+str(additional_comments)

    np.savetxt(save_featureWeights, featureWeights, header=header, delimiter=",", comments="")
    print("\n# Feature weights saved to file "+str(save_featureWeights))
    print("\n#### Done ####")
    return featureWeights
