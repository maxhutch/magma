/* 
    -- MAGMA (version 1.6.1) -- 
       Univ. of Tennessee, Knoxville 
       Univ. of California, Berkeley 
       Univ. of Colorado, Denver 
       May 2013 
 
       @author: Simplice Donfack 
 
*/

#include "magma_amc_args.h" /* Context and arguments */

int magma_amc_init(int P, double dcpu, int Pr, int nb){

    amc_args_t *args;
    
    /*create a new args*/
    args = magma_amc_args_create();

    /*initialize*/
    magma_amc_args_set(args, P, dcpu,  nb, Pr);

    /*set as default*/
    magma_amc_args_set_default(args);

    return 0;
}

int magma_amc_finalize(){
    amc_args_t *args;

    args= magma_amc_args_get_default();

    magma_amc_args_free(args);

    return 0;
}

