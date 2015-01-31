/* 
    -- MAGMA (version 1.6.1) -- 
       Univ. of Tennessee, Knoxville 
       Univ. of California, Berkeley 
       Univ. of Colorado, Denver 
       May 2013 
 
       @author: Simplice Donfack 
*/
#include <stdlib.h>
#include "magma_amc_args.h"

/* default args : used by default when args is not specified*/
/* A new args object may be used to enable thread safety*/
static amc_args_t *default_args;

/* Create a new context */
amc_args_t *magma_amc_args_create(){

    amc_args_t *args;

    args = (amc_args_t*) malloc(sizeof(amc_args_t));

    args->nb = 0;
    args->P = 2;
    args->Pr = 0;
    args->dcpu = 0;

    return args;
}

void magma_amc_args_set(amc_args_t *args, int P, double dcpu,  int nb, int Pr){
    
    args->P = P;
    args->Pr = Pr;
    args->nb = nb;
    args->dcpu = dcpu;
}

/**/
void magma_amc_args_free(amc_args_t *args){
    free(args);
}

/* set the default args */
void magma_amc_args_set_default(amc_args_t *args){
    default_args = args;
}

/* get default args */
amc_args_t *magma_amc_args_get_default(){
return default_args;
}


