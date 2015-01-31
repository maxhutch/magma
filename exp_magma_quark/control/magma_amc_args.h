/* 
    -- MAGMA (version 1.6.1) -- 
       Univ. of Tennessee, Knoxville 
       Univ. of California, Berkeley 
       Univ. of Colorado, Denver 
       May 2013 
 
       @author: Simplice Donfack 
 
*/

#ifndef MAGMA_AMC_ARGS_H
#define MAGMA_AMC_ARGS_H
/* Create a struct for setting context */
struct amc_args_struct {

 /*number of processors*/
 int P; 
 /*percentage of the matrix for the CPU*/ 
 double dcpu; 
 /*Block size*/
 int nb;
 /*Number of threads in the first dimension (exple: the panel number of threads)*/
 int Pr;
};

typedef struct amc_args_struct amc_args_t;

/* Create a new context */
amc_args_t *magma_amc_args_create();

void magma_amc_args_set(amc_args_t *args, int P, double dcpu,  int nb, int Pr);

void magma_amc_args_free(amc_args_t *args);

/* set the default args */
void magma_amc_args_set_default(amc_args_t *args);

/* get default args */
amc_args_t *magma_amc_args_get_default();
#endif

