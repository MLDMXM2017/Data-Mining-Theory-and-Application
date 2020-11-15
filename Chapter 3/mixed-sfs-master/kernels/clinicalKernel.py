import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda
import time

from pycuda.compiler import SourceModule

MAX_THREADS_PER_BLOCK = 1024

maxmin_mod = SourceModule("""
// Each column of the data runs on one blockX. Per block - one thread for max, one for min.
__global__ void maxmin_kernel(float *d_in, int *numRows, float *max_min){
__shared__ float max;
__shared__ float min;
    if(threadIdx.x == 0){
        // Compute the MAX value of the column
        float tempMax = d_in[blockIdx.x];
        for(int iter = 0; iter < *numRows; iter++){
            tempMax = fmaxf(tempMax, d_in[blockIdx.x + gridDim.x*iter]);
        }
        max = tempMax;
    }
    else if(threadIdx.x == 1){
        // Compute the MIN value of the column
        float tempMin = d_in[blockIdx.x];
        for(int iter = 0; iter < *numRows; iter++){
            tempMin = fminf(tempMin, d_in[blockIdx.x + gridDim.x*iter]);
        }
        min = tempMin;
    }
    __syncthreads();
    max_min[blockIdx.x] = max - min;
}
""")

normalize_mod = SourceModule("""
// Each row of data on one block. Per block - one thread per column.
__global__ void normalize_kernel(float *max_minus_min, float *d_in, float *d_out){
    int iter = threadIdx.x + blockDim.x*blockIdx.x;
    d_out[iter] = fdividef(d_in[iter], max_minus_min[threadIdx.x]);
}
""")


similarity_numerical_mod = SourceModule("""
__global__ void similarity_numerical_kernel(float *d, float *sim){
    __shared__ float similarity_accumulator_numerical;
    if(threadIdx.x == 0){
        // Initialize the similarity_accumulator_numerical to zero
        similarity_accumulator_numerical = 0;
    }
    __syncthreads(); //Wait until similarity_accumulator_numerical is initialized

    if(blockIdx.x > blockIdx.y){
        float tempSim = 1-abs(__fsub_rn(d[blockDim.x*blockIdx.x + threadIdx.x], d[blockDim.x*blockIdx.y + threadIdx.x]));
        atomicAdd(&similarity_accumulator_numerical, tempSim);
        __syncthreads(); //Wait until all threads have contributed to similarity_accumulator_numerical
        // Copy data to global memory only once per block, when threadIdx.x == 0
            sim[blockIdx.x+blockIdx.y*gridDim.x] = similarity_accumulator_numerical; //Write accumulated similarity to the array in global memory
            sim[blockIdx.y+blockIdx.x*gridDim.y] = similarity_accumulator_numerical;
    }
    else{
    }

}
""")

similarity_full_mod = SourceModule("""
__global__ void similarity_full_kernel(int *d, float *sim_numerical, float *numCols, float *sim){
    __shared__ int similarity_accumulator_categorical;
    if(threadIdx.x == 0){
        // Initialize the similarity_accumulator_categorical to zero
        similarity_accumulator_categorical = 0;
    }
    __syncthreads(); //Wait until similarity_accumulator_categorical is initialized

    if(blockIdx.x > blockIdx.y){
        bool tempSim = (d[blockDim.x*blockIdx.x+threadIdx.x] == d[blockDim.x*blockIdx.y+ threadIdx.x]);
        atomicAdd(&similarity_accumulator_categorical, int(tempSim));
        __syncthreads(); //Wait until all threads have contributed to similarity_accumulator
        if(threadIdx.x == 0){
        // Add similarity_accumulator_categorical to sim_numerical once per block, when threadIdx.x == 0
            float tempFinalSim = fdividef((__int2float_rd(similarity_accumulator_categorical) + sim_numerical[blockIdx.x+blockIdx.y*gridDim.x]), *numCols);
            sim[blockIdx.x+blockIdx.y*gridDim.x] = tempFinalSim; //Write total similarity (categorical and numerical) to the array in global memory
            // __int2float_rd(similarity_accumulator_categorical)
            sim[blockIdx.y+blockIdx.x*gridDim.y] = tempFinalSim;
        }
    }
    else{
    }
}
""")

div_by_numCols_mod = SourceModule("""
__global__ void div_by_numCols_kernel(float *simMatrix, int *numRows, float *numCols){
    int iter = blockIdx.x*blockDim.x + threadIdx.x;
    if(iter < ((*numRows)*(*numRows))){
        simMatrix[iter] = fdividef(simMatrix[iter], *numCols);
    }
}
""")

## Getting the functions
similarity_numerical_func = similarity_numerical_mod.get_function("similarity_numerical_kernel")

similarity_full_func = similarity_full_mod.get_function("similarity_full_kernel")

maxmin_func = maxmin_mod.get_function("maxmin_kernel")

normalize_func = normalize_mod.get_function("normalize_kernel")

div_by_numCols_func = div_by_numCols_mod.get_function("div_by_numCols_kernel")
##END Getting the functions
exec_time = 0

def clinicalKernel(data, catCols = np.array([], dtype=int), return_to_CPU = False):

    ### Input:
    ### # data: (numpy.ndarray) Input data for which clinicalKernel is to be computed
    ### # catCols: (numpy.ndarray, dtype=int) 1-D array containing indices of categorical columns in data
    ### # return_to_CPU: (bool) Input must be True if the output similarity matrix is required to be returned to CPU
    ###
    ### Output:
    ### # similarity matrix: (numpy.ndarray OR pycuda.driver.DeviceAllocation) If return_to_CPU = True, a numppy array is returned, else a pycuda.driver.DeviceAllocation object corresponding to the similarity matrix on GPU is returned.
    ###
    ### Description:
    ### # Compute the clinical kernel of mixed data (i.e. data containing both numerical and categorical data).
    ### # Clinical kernel is pbtained by a summation of subkernels over each feature. Refer: A. Daemen and B. De Moor, "Development of a kernel function for clinical data," 2009 Annual International Conference of the IEEE Engineering in Medicine and Biology Society, Minneapolis, MN, 2009, pp. 5913-5917 (https://ieeexplore.ieee.org/document/5334847)
    ### # The subkernels are computed using Heterogeneous Euclidian Overlap Metrix (HEOM). Ref: D.R. Wilson, T.R. Martinez, Improved heterogeneous distance functions, J. Artif. Intell. Res. 6 (1997) 1–34
    ### This is also described in Ref. Saúl Solorio-Fernández, José Fco. Martínez-Trinidad, J. Ariel Carrasco-Ochoa, A new Unsupervised Spectral Feature Selection Method for mixed data: A filter approach, Pattern Recognition, Volume 72, 2017, Pages 314-326,
    ### This is alsom implemented (wihout GPU acceleration) by scikit-survival package: sksurv.kernels.ClinicalKernelTransform

    start_time = time.time() ### Start clock for timing GPU execution
    print("# clinicalKernel running on GPU ...")
    data_only_numerical = (catCols.size == 0)
    data_only_categorical = (catCols.size == data.shape[1])
    if (np.any(catCols > data.shape[1]-1)):
        raise ValueError("Input 'catCols' contains column indices not present in input 'data'.")

    ### Very important
    # PyCUDA assumes that all arrays being sent to the GPU are in row-major order, aka C-style array. But, NumPy sometimes seems to be changing the order of an array, when sliced, or when other operations are performed. So we explicitly change the order to 'C' using np.asanyarray(array, order='C') before sending arrays to GPU.
    data = np.asanyarray(data, order='C', dtype=np.float32)
    numRows = data.shape[0]
    numCols = data.shape[1]

    if (not data_only_categorical):
        data_numerical = np.asanyarray(np.delete(data, catCols, axis=1), dtype=np.float32, order='C')
        numCols_num = data_numerical.shape[1]
        data_numerical_gpu = cuda.mem_alloc(data_numerical.nbytes)
        cuda.memcpy_htod(data_numerical_gpu, data_numerical)

    if (not data_only_numerical):
        data_categorical = data[:,catCols]
        data_categorical = np.asanyarray(data_categorical, dtype=np.int32, order='C')
        data_categorical_gpu = cuda.mem_alloc(data_categorical.nbytes)
        cuda.memcpy_htod(data_categorical_gpu, data_categorical)
        numCols_cat = data_categorical.shape[1]

    ######### Running maxmin_func #########
    threadsPerBlock = 2
    grid = (numCols_num, 1)
    block = (threadsPerBlock, 1, 1)

    max_minus_min = np.zeros(numCols_num, dtype=np.float32, order='C')
    max_minus_min_gpu = cuda.mem_alloc(max_minus_min.nbytes)
    numRows_32 = np.int32(numRows)
    numRows_gpu = cuda.mem_alloc(numRows_32.nbytes)
    cuda.memcpy_htod(numRows_gpu, numRows_32)

    maxmin_func.prepare([np.intp, np.intp, np.intp])
    maxmin_func.prepared_call(grid, block, data_numerical_gpu, numRows_gpu, max_minus_min_gpu)

#    cuda.memcpy_dtoh(max_minus_min, max_minus_min_gpu)
    #########END Running maxmin_func #########



    ######### Running normalize_func #########
    threadsPerBlock = numCols_num
    grid = (numRows, 1)
    block = (threadsPerBlock, 1, 1)

    data_numerical_norm = np.zeros((numRows, numCols_num), dtype=np.float32, order='C')
    data_numerical_norm_gpu = cuda.mem_alloc(data_numerical_norm.nbytes)

    normalize_func.prepare([np.intp, np.intp, np.intp])
    normalize_func.prepared_call(grid, block, max_minus_min_gpu, data_numerical_gpu, data_numerical_norm_gpu)
#    cuda.memcpy_dtoh(data_numerical_norm, data_numerical_norm_gpu)
    # Free up GPU memory by deleting max_minus_min_gpu
    max_minus_min_gpu.free()
    data_numerical_gpu.free()
    #########END Running normalize_func #########



    ######### Running similarity_numerical_func #########
    if (not data_only_categorical):
        threadsPerBlock = numCols_num
        grid = (numRows, numRows)
        block = (threadsPerBlock, 1, 1)

        similarity_numerical = np.zeros((numRows, numRows), dtype=np.float32, order='C')
        similarity_numerical_gpu = cuda.mem_alloc(similarity_numerical.nbytes)
        cuda.memcpy_htod(similarity_numerical_gpu, similarity_numerical)

        similarity_numerical_func.prepare([np.intp, np.intp])
        similarity_numerical_func.prepared_call(grid, block, data_numerical_norm_gpu, similarity_numerical_gpu)
        global exec_time
        if(data_only_numerical):
            if(return_to_CPU):
                # Divide similarity_numerical by numCols_num to get the final similarity matrix
                # We use div_by_numCols_kernel for this
                threadsPerBlock = MAX_THREADS_PER_BLOCK
                block = (threadsPerBlock, 1, 1)
                grid = (int((numRows + threadsPerBlock - 1)/threadsPerBlock), 1)
                numCols_num_gpu = cuda.mem_alloc(np.float32(numCols_num).nbytes)
                cuda.memcpy_htod(numCols_num_gpu, np.float32(numCols_num))
                div_by_numCols_func.prepare([np.intp, np.intp, np.intp])
                div_by_numCols_func.prepared_call(grid, block, similarity_numerical_gpu, numRows_gpu, numCols_num_gpu)
                cuda.memcpy_dtoh(similarity_numerical, similarity_numerical_gpu)
                exec_time = time.time() - start_time
                print("# Input contains no categorical data")
                print("# Computed similarity matrix (GPU exec time: "+time.strftime("%H:%M:%S", time.gmtime(exec_time))+")")
                print("# Output returned to CPU")
                ## Free up GPU memory
                data_numerical_norm_gpu.free()
                similarity_numerical_gpu.free()
                return similarity_numerical
            else:
                threadsPerBlock = MAX_THREADS_PER_BLOCK
                block = (threadsPerBlock, 1, 1)
                grid = (int((numRows + threadsPerBlock - 1)/threadsPerBlock), 1)
                numCols_num_gpu = cuda.mem_alloc(np.float32(numCols_num).nbytes)
                cuda.memcpy_htod(numCols_num_gpu, np.float32(numCols_num))
                div_by_numCols_func.prepare([np.intp, np.intp, np.intp])
                div_by_numCols_func.prepared_call(grid, block, similarity_numerical_gpu, numRows_gpu, numCols_num_gpu)
                exec_time = time.time() - start_time
                print("# Input contains no categorical data")
                print("# Computed similarity matrix (GPU exec time: "+time.strftime("%H:%M:%S", time.gmtime(exec_time))+")")
                exec_time = time.time() - start_time
                print("# Output stored on GPU")
                ##Free up GPU memory
                data_numerical_norm_gpu.free()
                return similarity_numerical_gpu
    #########END Running similarity_numerical_func #########



    ######### Running similarity_full_func #########
    threadsPerBlock = numCols_cat
    grid = (numRows, numRows)
    block = (threadsPerBlock, 1, 1)

    similarity_full = np.zeros((numRows, numRows), dtype=np.float32, order='C')
    similarity_full_gpu = cuda.mem_alloc(similarity_full.nbytes)
    cuda.memcpy_htod(similarity_full_gpu, similarity_full)
    numCols_float = np.float32(numCols)
    numCols_gpu = cuda.mem_alloc(numCols_float.nbytes)
    cuda.memcpy_htod(numCols_gpu, numCols_float)

    similarity_full_func.prepare([np.intp, np.intp, np.intp, np.intp])

    if data_only_categorical:
        similarity_numerical = np.zeros((numRows, numCols), dtype=np.float32, order='C')
        similarity_numerical_gpu = cuda.mem_alloc(similarity_numerical.nbytes)
        cuda.memcpy_htod(similarity_numerical_gpu, similarity_numerical)
        print("# Input contains no numerical data")

    similarity_full_func.prepared_call(grid, block, data_categorical_gpu, similarity_numerical_gpu, numCols_gpu, similarity_full_gpu)
    ## Free up GPU memory
    similarity_numerical_gpu.free()
    data_categorical_gpu.free()
    numCols_gpu.free()
    if(return_to_CPU):
        cuda.memcpy_dtoh(similarity_full, similarity_full_gpu)
        exec_time = time.time() - start_time
        print("# Computed similarity matrix (GPU exec time: "+time.strftime("%H:%M:%S", time.gmtime(exec_time))+")")
        ## Free up GPU memory
        similarity_full_gpu.free()
        print("# Output returned to CPU")
        return similarity_full
    else:
        exec_time = time.time() - start_time
        print("# Computed similarity matrix (GPU exec time: "+time.strftime("%H:%M:%S", time.gmtime(exec_time))+")")
        print("# Output stored on GPU")
        return similarity_full_gpu
    #########END Running similarity_full_func #########

def clinicalKernel_CPU(a, catCols):
    print("clinicalKernel_CPU ...")
    start_time_cpu = time.time()
    sm_cpu = np.zeros((a.shape[0], a.shape[0]), dtype=np.float32, order='C')
    a_num = np.delete(a, catCols, axis=1)
    a_cat = a[:, catCols]
    max_min_cpu = np.amax(a_num, axis=0)-np.amin(a_num, axis=0)
    mmcpu = np.tile(max_min_cpu, a.shape[0]).reshape(a_num.shape[0], a_num.shape[1])
    a_num_norm = np.divide(a_num, mmcpu)

    numRows = a.shape[0]
    for i in range(numRows):
        for j in range(numRows):
            tempSim = 0
            print(repr(i)+", "+repr(j), end='\r')
            for k in range(a_num.shape[1]):
                tempSim += 1-np.abs(a_num_norm[i,k] - a_num_norm[j,k])
            sm_cpu[i,j] = tempSim
    for i in range(numRows):
        for j in range(numRows):
            tempSimCat = 0
            for k in range(a_cat.shape[1]):
                tempSimCat += int(a_cat[i, k] == a_cat[j, k])
            sm_cpu[i,j] += tempSimCat

    global exec_time_cpu
    sm_cpu = sm_cpu/(a.shape[1])
    exec_time_cpu = round(time.time()-start_time_cpu, 3)
    if(catCols.size == 0):
        print("# Input contains no categorical data")
    print("# Computed similarity matrix on CPU (exec time: "+time.strftime("%H:%M:%S", time.gmtime(exec_time_cpu))+")")
    return sm_cpu

#### Functino to check whether GPU output and CPU output match element-wise, with specified tolerance, and to check execution timing
def GPUvsCPU(data, catCols, tolerance):
    print("\n###############\n")
    print("# GPUvsCPU: Check whether GPU and CPU outputs match, and compare timing\n")
    print("### GPU ...")
    gpu_kernel = clinicalKernel(data, catCols)
    print("\n### CPU ...")
    cpu_kernel = clinicalKernel_CPU(data, catCols)
    diff = np.abs(np.subtract(gpu_kernel, cpu_kernel))
    no_mismatch_elements = diff < tolerance
    correct = (no_mismatch_elements).all()
    print("\n###############\n")
    print("# GPU and CPU match? -- "+repr(correct))
    if(not correct):
        num_different_elements = np.sum((~no_mismatch_elements).astype(int))
        print("### Mismatch between CPU and GPU ###")
        print("# Outputs differ in "+repr(num_different_elements)+" matrix elements")
        print("# Input tolerance: "+repr(tolerance))
        print("# Mismatch: "+repr(round(100*num_different_elements/(data.shape[0]**2), 2))+" %")
        print("# Try with a higher tolerance?")
    print("# GPU speedup factor: "+repr(round(exec_time_cpu/exec_time, 2))+"x")
