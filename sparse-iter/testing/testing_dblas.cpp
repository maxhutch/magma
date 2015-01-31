/*
    -- MAGMA (version 1.6.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date January 2015

       @generated from testing_zblas.cpp normal z -> d, Fri Jan 30 19:00:33 2015
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
    
    double one = MAGMA_D_MAKE( 1.0, 0.0 );
    double two = MAGMA_D_MAKE( 2.0, 0.0 );

    magma_d_vector a, ad, bd, cd;
    magma_d_vinit( &a, Magma_CPU, n, one, queue );
    magma_d_vinit( &bd, Magma_DEV, n, two, queue );
    magma_d_vinit( &cd, Magma_DEV, n, one, queue );
    
    magma_d_vtransfer( a, &ad, Magma_CPU, Magma_DEV, queue ); 

    double res;
    res = magma_dnrm2(n, ad.dval, 1); 
    
    printf("res: %f\n", res);
    magma_dscal( n, two, ad.dval, 1 );   

    magma_daxpy( n, one, ad.dval, 1, bd.dval, 1 );
    
    magma_dcopy( n, bd.dval, 1, ad.dval, 1 );

    res = MAGMA_D_REAL( magma_ddot(n, ad.dval, 1, bd.dval, 1) );

    printf("res: %f\n", res);


    magma_d_vfree( &a, queue);
    magma_d_vfree(&ad, queue);
    magma_d_vfree(&bd, queue);
    magma_d_vfree(&cd, queue);

    /* Shutdown */
    magma_queue_destroy( queue );
    magma_finalize();
    
    return 0;
}
