/* 
    -- MAGMA (version 1.4.1) -- 
       Univ. of Tennessee, Knoxville 
       Univ. of California, Berkeley 
       Univ. of Colorado, Denver 
       May 2013 
 
       @author: Simplice Donfack 
 
       @generated d Thu May 23 11:46:05 2013 
 
*/
#ifndef MAGMA_ASYNC_ARGS_H
#define MAGMA_ASYNC_ARGS_H
/* Create a struct for setting context */
struct async_args_struct {

 /*number of processors*/
 int P; 
 /*percentage of the matrix for the CPU*/ 
 double dcpu; 
 /*Block size*/
 int nb;
 /*Number of threads in the first dimension (exple: the panel number of threads)*/
 int Pr;
};

typedef struct async_args_struct async_args_t;

/* Create a new context */
async_args_t *magma_async_args_create();

void magma_async_args_set(async_args_t *args, int P, double dcpu,  int nb, int Pr);

void magma_async_args_free(async_args_t *args);

/* set the default args */
void magma_async_args_set_default(async_args_t *args);

/* get default args */
async_args_t *magma_async_args_get_default();
#endif
