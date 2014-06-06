/* 
    -- MAGMA (version 1.4.1) -- 
       Univ. of Tennessee, Knoxville 
       Univ. of California, Berkeley 
       Univ. of Colorado, Denver 
       May 2013 
 
       @author: Simplice Donfack 
 
       @generated d Thu May 23 11:46:05 2013 
 
*/
#include <stdlib.h>
#include "magma_async_args.h"

/* default args : used by default when args is not specified*/
/* A new args object may be used to enable thread safety*/
static async_args_t *default_args;

/* Create a new context */
async_args_t *magma_async_args_create(){

    async_args_t *args;

    args = (async_args_t*) malloc(sizeof(async_args_t));

    args->nb = 0;
    args->P = 2;
    args->Pr = 0;
    args->dcpu = 0;

    return args;
}

void magma_async_args_set(async_args_t *args, int P, double dcpu,  int nb, int Pr){
    
    args->P = P;
    args->Pr = Pr;
    args->nb = nb;
    args->dcpu = dcpu;
}

/**/
void magma_async_args_free(async_args_t *args){
    free(args);
}

/* set the default args */
void magma_async_args_set_default(async_args_t *args){
    default_args = args;
}

/* get default args */
async_args_t *magma_async_args_get_default(){
return default_args;
}


