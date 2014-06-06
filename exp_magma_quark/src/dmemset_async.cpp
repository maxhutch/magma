/* 
    -- MAGMA (version 1.4.1) -- 
       Univ. of Tennessee, Knoxville 
       Univ. of California, Berkeley 
       Univ. of Colorado, Denver 
       May 2013 
 
       @author: Simplice Donfack 
 
*/
#include "common_magma.h"

#include "magma_async.h"

#include "schedule.h"

/*Initialize the matrix in parallel. This may prevent to use numactl --interleave*/
void magma_async_dmemset(double *ptr, double value, int n,  magma_int_t chunck, int P){

    int i, nsize, info;

    /* Check arguments */ 
    info = 0; 
    if (n < 0) 
        info = -1; 
    else if (chunck < 0) 
        info = -2; 
 
    if (info != 0) { 
        magma_xerbla( __func__, -info ); 
        return ; 
    } 
 
    /* Quick return if possible */ 
    if (n == 0 || chunck == 0) 
        return;
    
    /*Initialize the scheduler*/ 
     schedule_init(P);

    for(i=0;i<n;i+=chunck){
        nsize = min(n-i, chunck);
        schedule_insert_dmemset(&ptr[i], value, nsize);
    }

    /*Wait for all thread termination*/
    schedule_barrier();

    /*Shutdown the scheduler*/
     schedule_delete();
}
