/* 
    -- MAGMA (version 1.4.1) -- 
       Univ. of Tennessee, Knoxville 
       Univ. of California, Berkeley 
       Univ. of Colorado, Denver 
       May 2013 
 
       @author: Simplice Donfack 
 
*/
/*The routines control the initialisation of the routines called in magma_async*/
#ifndef MAGMA_ASYNC_CONTROLS_H
#define MAGMA_ASYNC_CONTROLS_H

int magma_async_init(int P, double dcpu, int Pr, int nb);
int magma_async_finalize();
#endif
