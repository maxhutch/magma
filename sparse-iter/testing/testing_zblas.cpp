/*
    -- MAGMA (version 1.6.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date January 2015

       @precisions normal z -> c d s
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
    
    magmaDoubleComplex one = MAGMA_Z_MAKE( 1.0, 0.0 );
    magmaDoubleComplex two = MAGMA_Z_MAKE( 2.0, 0.0 );

    magma_z_vector a, ad, bd, cd;
    magma_z_vinit( &a, Magma_CPU, n, one, queue );
    magma_z_vinit( &bd, Magma_DEV, n, two, queue );
    magma_z_vinit( &cd, Magma_DEV, n, one, queue );
    
    magma_z_vtransfer( a, &ad, Magma_CPU, Magma_DEV, queue ); 

    double res;
    res = magma_dznrm2(n, ad.dval, 1); 
    
    printf("res: %f\n", res);
    magma_zscal( n, two, ad.dval, 1 );   

    magma_zaxpy( n, one, ad.dval, 1, bd.dval, 1 );
    
    magma_zcopy( n, bd.dval, 1, ad.dval, 1 );

    res = MAGMA_Z_REAL( magma_zdotc(n, ad.dval, 1, bd.dval, 1) );

    printf("res: %f\n", res);


    magma_z_vfree( &a, queue);
    magma_z_vfree(&ad, queue);
    magma_z_vfree(&bd, queue);
    magma_z_vfree(&cd, queue);

    /* Shutdown */
    magma_queue_destroy( queue );
    magma_finalize();
    
    return 0;
}
