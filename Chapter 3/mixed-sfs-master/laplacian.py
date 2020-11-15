import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda
import time
from pycuda.compiler import SourceModule


degree_reciprocal_sqrt_mod = SourceModule("""
__global__ void degree_reciprocal_sqrt_kernel(float *simMatrix, int *numRows, float *degreeReciprocalSqrtArray){
    // threadsPerBlock is arbitrary. Each thread computes a single row sum.
    float rowSum = 0;
    int iter = blockIdx.x*blockDim.x + threadIdx.x;
    if(iter < (*numRows)){
        for(int iter2 = 0; iter2 < *numRows; iter2++){
            rowSum = rowSum + (simMatrix[*numRows*iter + iter2]);
        }
        degreeReciprocalSqrtArray[iter] = __frsqrt_rn(rowSum);
        //__frsqrt_rn(x) is the single precision intrinsic for 1/sqrt(x). Ref. CUDA Math API.
    }
}
""")

laplacian_mod = SourceModule("""
__global__ void laplacian_kernel(float *simMatrix, float *degreeReciprocalSqrtArray, int *numRows, float *laplacian_matrix){
    // 2D grid, gridDim.y = numRows, gridDim.x = (numRows + threadsPerBlock - 1)/threadsPerBlock
    // each row of simMatrix assigned to a row of blocks, containing threadsPerBlock threads
    int iter2 = (blockIdx.x*blockDim.x) + threadIdx.x;  //Iterate in a row
    int iter1 = (*numRows)*blockIdx.y + iter2;  //Global iterator
    if(iter2 < *numRows){
        if(blockIdx.y == iter2){
            //Diagonal elements
            laplacian_matrix[iter1] = 1 - degreeReciprocalSqrtArray[iter2]*simMatrix[iter1]*degreeReciprocalSqrtArray[iter2];
        }
        else {
            laplacian_matrix[iter1] = 0 -degreeReciprocalSqrtArray[blockIdx.y]*simMatrix[iter1]*degreeReciprocalSqrtArray[iter2];
        }
    }
}
""")

## Getting functions
degree_reciprocal_sqrt_func = degree_reciprocal_sqrt_mod.get_function("degree_reciprocal_sqrt_kernel")

laplacian_func = laplacian_mod.get_function("laplacian_kernel")

def laplacian_normalised(simMatrix, numRows, simMatrix_on_CPU=False, return_to_CPU=False):
    ### Input:
    ### # simMatrix: (numpy.ndarray) 2-D array of shape (numRows, numRows)
    ### # numRows: (int) number of rows in simMatrix. Passed additionally, because if simMatrix is on GPU, numRows cannot be inferred directly.
    ### #simMatrix_on_CPU: (bool) Input must be True if simMatrix is on CPU. Default: False
    ### #return_to_CPU: (bool) Input must be True if laplacian_matrix output must be returned to CPU.
    ###
    ### Output:
    ### # laplacian_matrix: (numpy.ndarray OR pycuda.driver.DeviceAllocation) If return_to_CPU = True, a 2-D numpy array is returned. If return_to_CPU = False, a pycuda.driver.DeviceAllocation object for the laplacian_matrix_gpu is returned.
    ###
    ### Description:
    ### # Computes the normalized Laplacian matrix corresponding to the ipnut similarity matrix.
    ### # Let A be the similarity matrix (a.k.a adjacency matrix), and D be the corresponding degree matrix. Then L = I - D^(-1/2).A.D^(-1/2) is the normalized laplacian of A.
    ### # Refer 1: https://en.wikipedia.org/wiki/Laplacian_matrix#Symmetric_normalized_Laplacian_2
    ### # Refer 2: https://people.orie.cornell.edu/dpw/orie6334/Fall2016/lecture7.pdf

    start_time = time.time()
    print("\n# laplacian_normalised running on GPU ...")
    if(simMatrix_on_CPU == True):
        simMatrix = np.asanyarray(simMatrix, dtype=np.float32, order='C')
        simMatrix_GPU = cuda.mem_alloc(simMatrix.nbytes)
        cuda.memcpy_htod(simMatrix_GPU, simMatrix)
    else:
        simMatrix_GPU = simMatrix
    numRows_32 = np.int32(numRows)
    numRows_gpu = cuda.mem_alloc(numRows_32.nbytes)
    cuda.memcpy_htod(numRows_gpu, numRows_32)

    #### Running degree_reciprocal_sqrt_func ####
    degreeReciprocalSqrtArray = np.zeros(numRows, dtype=np.float32, order='C')
    degreeReciprocalSqrtArray_gpu = cuda.mem_alloc(degreeReciprocalSqrtArray.nbytes)

    degree_reciprocal_sqrt_func.prepare([np.intp, np.intp, np.intp])
    threadsPerBlock = 256
    grid = (int((numRows + threadsPerBlock - 1)/threadsPerBlock), 1)
    block = (threadsPerBlock, 1 , 1)
    degree_reciprocal_sqrt_func.prepared_call(grid, block, simMatrix_GPU, numRows_gpu, degreeReciprocalSqrtArray_gpu)
    ####END Running degree_reciprocal_sqrt_func ####


    #### Running laplacian_func ####
    laplacian_matrix = np.zeros((numRows, numRows), dtype=np.float32, order='C')
    laplacian_matrix_gpu = cuda.mem_alloc(laplacian_matrix.nbytes)
    laplacian_func.prepare([np.intp, np.intp, np.intp, np.intp])

    threadsPerBlock = 256
    grid = (int((numRows + threadsPerBlock - 1)/threadsPerBlock), int(numRows))
    block = (threadsPerBlock, 1, 1)
    laplacian_func.prepared_call(grid, block, simMatrix_GPU, degreeReciprocalSqrtArray_gpu, numRows_gpu, laplacian_matrix_gpu)
    ## Free up GPU memory
    simMatrix_GPU.free()
    degreeReciprocalSqrtArray_gpu.free()

    if(return_to_CPU):
        # laplacian_matrix = np.zeros((numRows, numRows), dtype=np.float32, order='C')
        cuda.memcpy_dtoh(laplacian_matrix, laplacian_matrix_gpu)
        exec_time = time.time() - start_time
        print("# Computed Laplacian (GPU exec time: "+time.strftime("%H:%M:%S", time.gmtime(exec_time))+")")
        print("# Output returned to CPU")
        ## Free up GPU memory
        laplacian_matrix_gpu.free()
        return laplacian_matrix
    else:
        exec_time = time.time() - start_time
        print("# Computed Laplacian (GPU exec time: "+time.strftime("%H:%M:%S", time.gmtime(exec_time))+")")
        print("# Output stored on GPU")
        return laplacian_matrix_gpu
    ####END Running laplacian_func ####
