/*
    -- MAGMA (version 2.0.2) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date May 2016
       
       @generated from magmablas/zset_pointer.cu normal z -> s, Mon May  2 23:30:41 2016
       @author Azzam Haidar
       @author Tingxing Dong

*/

#include "magma_internal.h"
///////////////////////////////////////////////////////////////////////////////////////
__global__ void kernel_sset_pointer(
    float **output_array,
    float *input,
    magma_int_t lda,
    magma_int_t row, magma_int_t column, 
    magma_int_t batch_offset)
{
    output_array[blockIdx.x] =  input + blockIdx.x * batch_offset + row + column * lda;
    //printf("==> kernel_set_pointer input_array %p output_array %p  \n",input+ blockIdx.x * batch_offset,output_array[blockIdx.x]);
}

/*
   Purpose
    -------

    convert consecutive stored variable to array stored
    for example the size  of A is N*batchCount; N is the size of A(batch_offset)
    change into dA_array[0] dA_array[1],... dA_array[batchCount-1], where the size of each dA_array[i] is N
    
    Arguments
    ----------

    @param[out]
    output_array 	Array of pointers, dimension (batchCount).
             Each is a REAL array A of DIMENSION ( lda, column ) on the GPU
   
    @param[in]
    input      REAL array of dimension ( LDDA, N*batchCount ) on the GPU.


    @param[in]
    lda    INTEGER
            LDA specifies the leading dimension of A.

    @param[in]
    row       INTEGER
            On entry, row specifies the number of rows of the matrix A.

    @param[in]
    column       INTEGER
            On entry, column specifies the number of columns of the matrix A

    @param[in]
    batch_offset  INTEGER
                The starting pointer of each matrix A in input arrray

    @param[in]
    batchCount  INTEGER
                The number of matrices to operate on.

    @param[in]
    queue   magma_queue_t
            Queue to execute in.

*/

extern "C"
void magma_sset_pointer(
    float **output_array,
    float *input,
    magma_int_t lda,
    magma_int_t row, magma_int_t column, 
    magma_int_t batch_offset,
    magma_int_t batchCount, 
    magma_queue_t queue)
{
    kernel_sset_pointer
        <<< batchCount, 1, 0, queue->cuda_stream() >>>
        (output_array, input, lda,  row, column, batch_offset);
}

//////////////////////////////////////////////////////////////////////////////////////////


__global__ void zdisplace_pointers_kernel(float **output_array,
               float **input_array, magma_int_t lda,
               magma_int_t row, magma_int_t column)
{
    float *inpt = input_array[blockIdx.x];
    output_array[blockIdx.x] = &inpt[row + column * lda];
}


/*
   Purpose
    -------

    compute the offset for all the matrices and save the displacment of the new pointer on output_array.
    input_array contains the pointers to the initial position.
    output_array[i] = input_array[i] + row + lda * column; 
    
    Arguments
    ----------

    @param[out]
    output_array 	Array of pointers, dimension (batchCount).
             Each pointer points to the new displacement of array A in input_array on the GPU
   
    @param[in]
    input_array 	Array of pointers, dimension (batchCount).
             Each is a REAL array A of DIMENSION ( lda, column ) on the GPU

    @param[in]
    lda    INTEGER
            LDA specifies the leading dimension of A.

    @param[in]
    row       INTEGER
            On entry, row specifies the number of rows of the matrix A.

    @param[in]
    column       INTEGER
            On entry, column specifies the number of columns of the matrix A

    @param[in]
    batch_offset  INTEGER
                The starting pointer of each matrix A in input arrray

    @param[in]
    batchCount  INTEGER
                The number of matrices to operate on.

    @param[in]
    queue   magma_queue_t
            Queue to execute in.

*/

extern "C"
void magma_sdisplace_pointers(float **output_array,
               float **input_array, magma_int_t lda,
               magma_int_t row, magma_int_t column, 
               magma_int_t batchCount, magma_queue_t queue)
{
    zdisplace_pointers_kernel
        <<< batchCount, 1, 0, queue->cuda_stream() >>>
        (output_array, input_array, lda, row, column);
}
