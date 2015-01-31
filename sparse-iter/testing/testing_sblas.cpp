/*
    -- MAGMA (version 1.6.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date January 2015

       @generated from testing_zblas.cpp normal z -> s, Fri Jan 30 19:00:33 2015
       @author Hartwig Anzt
*/

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

// includes, project
#include "flops.h"
#include "magma.h"
#include "magmasparse.h"
#include "magma_lapack.h"
#include "testings.h"



/* ////////////////////////////////////////////////////////////////////////////
   -- testing any solver 
*/
int main(  int argc, char** argv )
{
    /* Initialize */
    TESTING_INIT();
        magma_queue_t queue;
            magma_queue_create( /*devices[ opts->device ],*/ &queue );

    magma_int_t n=10000;
    
    float one = MAGMA_S_MAKE( 1.0, 0.0 );
    float two = MAGMA_S_MAKE( 2.0, 0.0 );

    magma_s_vector a, ad, bd, cd;
    magma_s_vinit( &a, Magma_CPU, n, one, queue );
    magma_s_vinit( &bd, Magma_DEV, n, two, queue );
    magma_s_vinit( &cd, Magma_DEV, n, one, queue );
    
    magma_s_vtransfer( a, &ad, Magma_CPU, Magma_DEV, queue ); 

    float res;
    res = magma_snrm2(n, ad.dval, 1); 
    
    printf("res: %f\n", res);
    magma_sscal( n, two, ad.dval, 1 );   

    magma_saxpy( n, one, ad.dval, 1, bd.dval, 1 );
    
    magma_scopy( n, bd.dval, 1, ad.dval, 1 );

    res = MAGMA_S_REAL( magma_sdot(n, ad.dval, 1, bd.dval, 1) );

    printf("res: %f\n", res);


    magma_s_vfree( &a, queue);
    magma_s_vfree(&ad, queue);
    magma_s_vfree(&bd, queue);
    magma_s_vfree(&cd, queue);

    /* Shutdown */
    magma_queue_destroy( queue );
    magma_finalize();
    
    return 0;
}
