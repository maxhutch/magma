/*
    -- MAGMA (version 1.6.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date January 2015

       @generated from testing_zmconverter.cpp normal z -> s, Fri Jan 30 19:00:33 2015
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

    magma_sopts zopts;
    magma_queue_t queue;
    magma_queue_create( /*devices[ opts->device ],*/ &queue );
    
    int i=1;
    magma_sparse_opts( argc, argv, &zopts, &i, queue );


    real_Double_t res;
    magma_s_sparse_matrix Z, Z2, A, A2, AT, AT2, B;

    B.blocksize = zopts.blocksize;
    B.alignment = zopts.alignment;

    while(  i < argc ) {

        if ( strcmp("LAPLACE2D", argv[i]) == 0 && i+1 < argc ) {   // Laplace test
            i++;
            magma_int_t laplace_size = atoi( argv[i] );
            magma_sm_5stencil(  laplace_size, &Z, queue );
        } else {                        // file-matrix test
            magma_s_csr_mtx( &Z,  argv[i], queue );
        }

        printf( "# matrix info: %d-by-%d with %d nonzeros\n",
                            (int) Z.num_rows,(int) Z.num_cols,(int) Z.nnz );
        
        // convert to be non-symmetric
        magma_s_mconvert( Z, &A, Magma_CSR, Magma_CSRL, queue );
        magma_s_mconvert( Z, &B, Magma_CSR, Magma_CSRU, queue );
        
        // transpose
        magma_s_mtranspose( A, &AT, queue );
        
        // quite some conversions
        
        //ELL
        magma_s_mconvert( AT, &AT2, Magma_CSR, Magma_ELL, queue );
        magma_s_mfree(&AT, queue );        
        magma_s_mconvert( AT2, &AT, Magma_ELL, Magma_CSR, queue );  
        magma_s_mfree(&AT2, queue );
        //ELLPACKT
        magma_s_mconvert( AT, &AT2, Magma_CSR, Magma_ELLPACKT, queue );
        magma_s_mfree(&AT, queue );        
        magma_s_mconvert( AT2, &AT, Magma_ELLPACKT, Magma_CSR, queue );  
        magma_s_mfree(&AT2, queue );
        //ELLRT
        AT2.blocksize = 8;
        AT2.alignment = 8;
        magma_s_mconvert( AT, &AT2, Magma_CSR, Magma_ELLRT, queue );
        magma_s_mfree(&AT, queue );        
        magma_s_mconvert( AT2, &AT, Magma_ELLRT, Magma_CSR, queue );  
        magma_s_mfree(&AT2, queue );
        //SELLP
        AT2.blocksize = 8;
        AT2.alignment = 8;
        magma_s_mconvert( AT, &AT2, Magma_CSR, Magma_SELLP, queue );
        magma_s_mfree(&AT, queue );   
        magma_s_mconvert( AT2, &AT, Magma_SELLP, Magma_CSR, queue );  
        magma_s_mfree(&AT2, queue );
        //ELLD
        magma_s_mconvert( AT, &AT2, Magma_CSR, Magma_ELLD, queue );
        magma_s_mfree(&AT, queue );        
        magma_s_mconvert( AT2, &AT, Magma_ELLD, Magma_CSR, queue );  
        magma_s_mfree(&AT2, queue );
        //CSRCOO
        magma_s_mconvert( AT, &AT2, Magma_CSR, Magma_CSRCOO, queue );
        magma_s_mfree(&AT, queue );        
        magma_s_mconvert( AT2, &AT, Magma_CSRCOO, Magma_CSR, queue );  
        magma_s_mfree(&AT2, queue );
        //CSRD
        magma_s_mconvert( AT, &AT2, Magma_CSR, Magma_CSRD, queue );
        magma_s_mfree(&AT, queue );        
        magma_s_mconvert( AT2, &AT, Magma_CSRD, Magma_CSR, queue );  
        magma_s_mfree(&AT2, queue );
        //BCSR
        magma_s_mconvert( AT, &AT2, Magma_CSR, Magma_BCSR, queue );
        magma_s_mfree(&AT, queue );        
        magma_s_mconvert( AT2, &AT, Magma_BCSR, Magma_CSR, queue );  
        magma_s_mfree(&AT2, queue );
        
        // transpose
        magma_s_mtranspose( AT, &A2, queue );
        
        magma_smdiff( A, A2, &res, queue);
        printf("# ||A-A2||_F = %8.2e\n", res);
        if ( res < .000001 )
            printf("# conversion tester:  ok\n");
        else
            printf("# conversion tester:  failed\n");
        
        magma_smlumerge( A2, B, &Z2, queue );

        
        magma_smdiff( Z, Z2, &res, queue);
        printf("# ||Z-Z2||_F = %8.2e\n", res);
        if ( res < .000001 )
            printf("# LUmerge tester:  ok\n");
        else
            printf("# LUmerge tester:  failed\n");
        


        magma_s_mfree(&A, queue ); 
        magma_s_mfree(&A2, queue );
        magma_s_mfree(&AT, queue ); 
        magma_s_mfree(&AT2, queue ); 
        magma_s_mfree(&B, queue ); 
        magma_s_mfree(&Z2, queue );
        magma_s_mfree(&Z, queue ); 

        i++;
    }
    magma_queue_destroy( queue );
    TESTING_FINALIZE();
    return 0;
}
