/* 
    -- MAGMA (version 1.4.1) -- 
       Univ. of Tennessee, Knoxville 
       Univ. of California, Berkeley 
       Univ. of Colorado, Denver 
       May 2013 
 
       @author: Simplice Donfack 
 
       @generated d Thu May 23 11:46:05 2013 
 
*/
#include "magma_async_args.h" /* Context and arguments */

int magma_async_init(int P, double dcpu, int Pr, int nb){

    async_args_t *args;
    
    /*create a new args*/
    args = magma_async_args_create();

    /*initialize*/
    magma_async_args_set(args, P, dcpu,  nb, Pr);

    /*set as default*/
    magma_async_args_set_default(args);

    return 0;
}

int magma_async_finalize(){
    async_args_t *args;

    args= magma_async_args_get_default();

    magma_async_args_free(args);

    return 0;
}
