/*
    -- MAGMA (version 1.6.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       November 2013
       
       @generated from zset_pointer.cu normal z -> c, Fri Jan 30 19:00:10 2015
       @author Azzam Haidar
       @author Tingxing Dong

*/

#include "common_magma.h"
///////////////////////////////////////////////////////////////////////////////////////
__global__ void kernel_set_pointer(magmaFloatComplex **output_array,
                 magmaFloatComplex *input,
                 magma_int_t lda,
                 magma_int_t row, magma_int_t column, 
                 magma_int_t batchSize)
{
     output_array[blockIdx.x] =  input + blockIdx.x * batchSize + row + column * lda;
     //printf("==> kernel_set_pointer input_array %p output_array %p  \n",input+ blockIdx.x * batchSize,output_array[blockIdx.x]);
}


extern "C"
void cset_pointer(magmaFloatComplex **output_array,
                 magmaFloatComplex *input,
                 magma_int_t lda,
                 magma_int_t row, magma_int_t column, 
                 magma_int_t batchSize,
                 magma_int_t batchCount, 
                 magma_queue_t queue)

{
/*
    convert consecutive stored variable to array stored
    for example the size  of A is N*batchCount; N is the size of A(batchSize)
    change into dA_array[0] dA_array[1],... dA_array[batchCount-1], where the size of each dA_array[i] is N
*/


    kernel_set_pointer<<<batchCount, 1, 0, queue>>>(output_array, input, lda,  row, column, batchSize);
}



__global__ void zdisplace_pointers_kernel(magmaFloatComplex **output_array,
               magmaFloatComplex **input_array, magma_int_t lda,
               magma_int_t row, magma_int_t column)
{
      magmaFloatComplex *inpt = input_array[blockIdx.x];
      output_array[blockIdx.x] = &inpt[row + column * lda];
}


extern "C"
void cset_array(magmaFloatComplex **output_array,
               magmaFloatComplex **input_array, magma_int_t lda,
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

extern "C"
void magma_cdisplace_pointers(magmaFloatComplex **output_array,
               magmaFloatComplex **input_array, magma_int_t lda,
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



