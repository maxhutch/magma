/*
    -- MAGMA (version 2.2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date November 2016
       
       @generated from magmablas/zset_pointer.cu, normal z -> s, Sun Nov 20 20:20:31 2016
       @author Azzam Haidar
       @author Tingxing Dong

*/

#include "magma_internal.h"

/******************************************************************************/
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
/******************************************************************************/
// set pointer with variable size matrices, batch_offset becomes an array with accumulated sum of sizes 
// batch_offset[i] = sum( matrix_size[0], matrix_size[1], ..., matrix_size[i-1])
// batch_offset is usually the output of a prefix sum operation
__global__ void kernel_sset_pointer_var(
    float **output_array,
    float *input,
    magma_int_t *lda,
    magma_int_t row, magma_int_t column, 
    magma_int_t *batch_offset)
{
    output_array[blockIdx.x] =  input + batch_offset[blockIdx.x] + row + column * lda[blockIdx.x];
    //printf("==> kernel_set_pointer input_array %p output_array %p  \n",input+ blockIdx.x * batch_offset,output_array[blockIdx.x]);
}


/***************************************************************************//**
    Purpose
    -------

    convert consecutive stored variable to array stored
    for example the size  of A is N*batchCount; N is the size of A(batch_offset)
    change into dA_array[0] dA_array[1],... dA_array[batchCount-1], where the size of each dA_array[i] is N
    
    Arguments
    ----------

    @param[out]
    output_array  Array of pointers, dimension (batchCount).
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
*******************************************************************************/
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
/******************************************************************************/
extern "C"
void magma_sset_pointer_var_cc(
    float **output_array,
    float *input,
    magma_int_t *lda,
    magma_int_t row, magma_int_t column, 
    magma_int_t *batch_offset,
    magma_int_t batchCount, 
    magma_queue_t queue)
{
    kernel_sset_pointer_var
        <<< batchCount, 1, 0, queue->cuda_stream() >>>
        (output_array, input, lda,  row, column, batch_offset);
}


/******************************************************************************/
__global__ void zdisplace_pointers_kernel(float **output_array,
               float **input_array, magma_int_t lda,
               magma_int_t row, magma_int_t column)
{
    float *inpt = input_array[blockIdx.x];
    output_array[blockIdx.x] = &inpt[row + column * lda];
}
/******************************************************************************/
/*  Variable pointer displacement kernels                                     */
/******************************************************************************/
// variable leading dimension, constant row and column offsets
__global__ void zdisplace_pointers_var_cc_kernel(float **output_array,
               float **input_array, magma_int_t* lda,
               magma_int_t row, magma_int_t column)
{
    const int bid = blockIdx.x;
    float *inpt = input_array[blockIdx.x];
    if(inpt == NULL || row < 0 || column < 0) 
        output_array[bid] = NULL;
    else
        output_array[bid] = &inpt[row + column * lda[blockIdx.x] ];
}
/******************************************************************************/
// variable leading dimension, constant row offset and variable column offsets
__global__ void zdisplace_pointers_var_cv_kernel(float **output_array,
               float **input_array, magma_int_t* lda,
               magma_int_t row, magma_int_t *column)
{
    const int bid = blockIdx.x;
    float *inpt = input_array[blockIdx.x];
    if(inpt == NULL || row < 0 || column[bid] < 0) 
        output_array[bid] = NULL;
    else
        output_array[bid] = &inpt[row + column[bid] * lda[blockIdx.x] ];
}
/******************************************************************************/
// variable leading dimension, variable row offset and  constant column offsets
__global__ void zdisplace_pointers_var_vc_kernel(float **output_array,
               float **input_array, magma_int_t* lda,
               magma_int_t *row, magma_int_t column)
{
    const int bid = blockIdx.x;
    float *inpt = input_array[blockIdx.x];
    if(inpt == NULL || row[bid] < 0 || column < 0) 
        output_array[bid] = NULL;
    else
        output_array[bid] = &inpt[row[bid] + column * lda[blockIdx.x] ];
}
/******************************************************************************/
// variable leading dimension, variable row and column offsets
__global__ void zdisplace_pointers_var_vv_kernel(float **output_array,
               float **input_array, magma_int_t* lda,
               magma_int_t* row, magma_int_t* column)
{
    const int bid = blockIdx.x;
    float *inpt = input_array[bid];
    if(inpt == NULL || row[bid] < 0 || column[bid] < 0) 
        output_array[bid] = NULL;  
    else
        output_array[bid] = &inpt[ row[bid] + column[bid] * lda[bid] ];
}

/***************************************************************************//**
    Purpose
    -------

    compute the offset for all the matrices and save the displacment of the new pointer on output_array.
    input_array contains the pointers to the initial position.
    output_array[i] = input_array[i] + row + lda * column; 
    
    Arguments
    ----------

    @param[out]
    output_array    Array of pointers, dimension (batchCount).
             Each pointer points to the new displacement of array A in input_array on the GPU
   
    @param[in]
    input_array     Array of pointers, dimension (batchCount).
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
*******************************************************************************/
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
/******************************************************************************/
extern "C"
void magma_sdisplace_pointers_var_cc(float **output_array,
               float **input_array, magma_int_t* lda,
               magma_int_t row, magma_int_t column, 
               magma_int_t batchCount, magma_queue_t queue)
{
/*
    compute the offset for all the matrices and save the displacment of the new pointer on output_array.
    input_array contains the pointers to the initial position.
    output_array[i] = input_array[i] + row + lda[i] * column; 
*/
    zdisplace_pointers_var_cc_kernel<<<batchCount, 1, 0, queue->cuda_stream()>>>(output_array, input_array, lda, row, column);
}
/******************************************************************************/
extern "C"
void magma_sdisplace_pointers_var_cv(float **output_array,
               float **input_array, magma_int_t* lda,
               magma_int_t row, magma_int_t* column, 
               magma_int_t batchCount, magma_queue_t queue)
{
/*
    compute the offset for all the matrices and save the displacment of the new pointer on output_array.
    input_array contains the pointers to the initial position.
    output_array[i] = input_array[i] + row + lda[i] * column[i]; 
*/
    zdisplace_pointers_var_cv_kernel<<<batchCount, 1, 0, queue->cuda_stream()>>>(output_array, input_array, lda, row, column);
}
/******************************************************************************/
extern "C"
void magma_sdisplace_pointers_var_vc(float **output_array,
               float **input_array, magma_int_t* lda,
               magma_int_t *row, magma_int_t column, 
               magma_int_t batchCount, magma_queue_t queue)
{
/*
    compute the offset for all the matrices and save the displacment of the new pointer on output_array.
    input_array contains the pointers to the initial position.
    output_array[i] = input_array[i] + row[i] + lda[i] * column; 
*/
    zdisplace_pointers_var_vc_kernel<<<batchCount, 1, 0, queue->cuda_stream()>>>(output_array, input_array, lda, row, column);
}
/******************************************************************************/
extern "C"
void magma_sdisplace_pointers_var_vv(float **output_array,
               float **input_array, magma_int_t* lda,
               magma_int_t* row, magma_int_t* column, 
               magma_int_t batchCount, magma_queue_t queue)
{
/*
    compute the offset for all the matrices and save the displacment of the new pointer on output_array.
    input_array contains the pointers to the initial position.
    output_array[i] = input_array[i] + row[i] + lda[i] * column[i]; 
*/
    zdisplace_pointers_var_vv_kernel<<<batchCount, 1, 0, queue->cuda_stream()>>>(output_array, input_array, lda, row, column);
}
