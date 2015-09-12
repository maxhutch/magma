/*
    -- MAGMA (version 1.7.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date September 2015
       
       @generated from zset_pointer.cu normal z -> s, Fri Sep 11 18:29:22 2015
       @author Azzam Haidar
       @author Tingxing Dong

*/

#include "common_magma.h"
///////////////////////////////////////////////////////////////////////////////////////
__global__ void kernel_set_pointer(float **output_array,
                 float *input,
                 magma_int_t lda,
                 magma_int_t row, magma_int_t column, 
                 magma_int_t batch_offset)
{
    output_array[blockIdx.x] =  input + blockIdx.x * batch_offset + row + column * lda;
    //printf("==> kernel_set_pointer input_array %p output_array %p  \n",input+ blockIdx.x * batch_offset,output_array[blockIdx.x]);
}


extern "C"
void sset_pointer(float **output_array,
                 float *input,
                 magma_int_t lda,
                 magma_int_t row, magma_int_t column, 
                 magma_int_t batch_offset,
                 magma_int_t batchCount, 
                 magma_queue_t queue)

{
    /*
    convert consecutive stored variable to array stored
    for example the size  of A is N*batchCount; N is the size of A(batch_offset)
    change into dA_array[0] dA_array[1],... dA_array[batchCount-1], where the size of each dA_array[i] is N
    */
    kernel_set_pointer<<<batchCount, 1, 0, queue>>>(output_array, input, lda,  row, column, batch_offset);
}



__global__ void zdisplace_pointers_kernel(float **output_array,
               float **input_array, magma_int_t lda,
               magma_int_t row, magma_int_t column)
{
    float *inpt = input_array[blockIdx.x];
    output_array[blockIdx.x] = &inpt[row + column * lda];
}


extern "C"
void magma_sdisplace_pointers(float **output_array,
               float **input_array, magma_int_t lda,
               magma_int_t row, magma_int_t column, 
               magma_int_t batchCount, magma_queue_t queue)

{
    /*
    compute the offset for all the matrices and save the displacment of the new pointer on output_array.
    input_array contains the pointers to the initial position.
    output_array[i] = input_array[i] + row + lda * column; 
    */
    zdisplace_pointers_kernel<<<batchCount, 1, 0, queue>>>(output_array, input_array, lda, row, column);
}
