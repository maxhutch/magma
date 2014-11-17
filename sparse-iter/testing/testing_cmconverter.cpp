/*
    -- MAGMA (version 1.6.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date November 2014

       @generated from testing_zmconverter.cpp normal z -> c, Sat Nov 15 19:54:24 2014
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
    TESTING_INIT();

    magma_copts zopts;
    magma_queue_t queue;
    magma_queue_create( /*devices[ opts->device ],*/ &queue );
    
    int i=1;
    magma_cparse_opts( argc, argv, &zopts, &i, queue );


    real_Double_t res;
    magma_c_sparse_matrix Z, Z2, A, A2, AT, AT2, B;

    B.blocksize = zopts.blocksize;
    B.alignment = zopts.alignment;

    while(  i < argc ) {

        if ( strcmp("LAPLACE2D", argv[i]) == 0 && i+1 < argc ) {   // Laplace test
            i++;
            magma_int_t laplace_size = atoi( argv[i] );
            magma_cm_5stencil(  laplace_size, &Z, queue );
        } else {                        // file-matrix test
            magma_c_csr_mtx( &Z,  argv[i], queue );
        }

        printf( "# matrix info: %d-by-%d with %d nonzeros\n",
                            (int) Z.num_rows,(int) Z.num_cols,(int) Z.nnz );
        
        // convert to be non-symmetric
        magma_c_mconvert( Z, &A, Magma_CSR, Magma_CSRL, queue );
        magma_c_mconvert( Z, &B, Magma_CSR, Magma_CSRU, queue );
        
        // transpose
        magma_c_mtranspose( A, &AT, queue );
        
        // quite some conversions
        
        //ELL
        magma_c_mconvert( AT, &AT2, Magma_CSR, Magma_ELL, queue );
        magma_c_mfree(&AT, queue );        
        magma_c_mconvert( AT2, &AT, Magma_ELL, Magma_CSR, queue );  
        magma_c_mfree(&AT2, queue );
        //ELLPACKT
        magma_c_mconvert( AT, &AT2, Magma_CSR, Magma_ELLPACKT, queue );
        magma_c_mfree(&AT, queue );        
        magma_c_mconvert( AT2, &AT, Magma_ELLPACKT, Magma_CSR, queue );  
        magma_c_mfree(&AT2, queue );
        //ELLRT
        AT2.blocksize = 8;
        AT2.alignment = 8;
        magma_c_mconvert( AT, &AT2, Magma_CSR, Magma_ELLRT, queue );
        magma_c_mfree(&AT, queue );        
        magma_c_mconvert( AT2, &AT, Magma_ELLRT, Magma_CSR, queue );  
        magma_c_mfree(&AT2, queue );
        //SELLP
        AT2.blocksize = 8;
        AT2.alignment = 8;
        magma_c_mconvert( AT, &AT2, Magma_CSR, Magma_SELLP, queue );
        magma_c_mfree(&AT, queue );   
        magma_c_mconvert( AT2, &AT, Magma_SELLP, Magma_CSR, queue );  
        magma_c_mfree(&AT2, queue );
        //ELLD
        magma_c_mconvert( AT, &AT2, Magma_CSR, Magma_ELLD, queue );
        magma_c_mfree(&AT, queue );        
        magma_c_mconvert( AT2, &AT, Magma_ELLD, Magma_CSR, queue );  
        magma_c_mfree(&AT2, queue );
        //CSRCOO
        magma_c_mconvert( AT, &AT2, Magma_CSR, Magma_CSRCOO, queue );
        magma_c_mfree(&AT, queue );        
        magma_c_mconvert( AT2, &AT, Magma_CSRCOO, Magma_CSR, queue );  
        magma_c_mfree(&AT2, queue );
        //CSRD
        magma_c_mconvert( AT, &AT2, Magma_CSR, Magma_CSRD, queue );
        magma_c_mfree(&AT, queue );        
        magma_c_mconvert( AT2, &AT, Magma_CSRD, Magma_CSR, queue );  
        magma_c_mfree(&AT2, queue );
        //BCSR
        magma_c_mconvert( AT, &AT2, Magma_CSR, Magma_BCSR, queue );
        magma_c_mfree(&AT, queue );        
        magma_c_mconvert( AT2, &AT, Magma_BCSR, Magma_CSR, queue );  
        magma_c_mfree(&AT2, queue );
        
        // transpose
        magma_c_mtranspose( AT, &A2, queue );
        
        magma_cmdiff( A, A2, &res, queue);
        printf("# ||A-A2||_F = %8.2e\n", res);
        if ( res < .000001 )
            printf("# conversion tester:  ok\n");
        else
            printf("# conversion tester:  failed\n");
        
        magma_cmlumerge( A2, B, &Z2, queue );

        
        magma_cmdiff( Z, Z2, &res, queue);
        printf("# ||Z-Z2||_F = %8.2e\n", res);
        if ( res < .000001 )
            printf("# LUmerge tester:  ok\n");
        else
            printf("# LUmerge tester:  failed\n");
        


        magma_c_mfree(&A, queue ); 
        magma_c_mfree(&A2, queue );
        magma_c_mfree(&AT, queue ); 
        magma_c_mfree(&AT2, queue ); 
        magma_c_mfree(&B, queue ); 
        magma_c_mfree(&Z2, queue );
        magma_c_mfree(&Z, queue ); 

        i++;
    }
    magma_queue_destroy( queue );
    TESTING_FINALIZE();
    return 0;
}
