/*
    -- MAGMA (version 1.5.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date September 2014

       @author Hartwig Anzt 

       @precisions normal z -> s d c
*/

#include "common_magma.h"
#include "magmasparse.h"

#include <assert.h>

#if defined(HAVE_PASTIX)
//PaStiX include
#include <stdint.h>
/* to access functions from the libpastix, respect this order */
#include <pastix.h>
#include <read_matrix.h>
#include <get_options.h>
#include <assert.h>
#define MPI_COMM_WORLD 0
#endif




#define PRECISION_z






/**
    Purpose
    -------

    Prepares the PASTIX solver

    Arguments
    ---------

    @param
    A           magma_z_sparse_matrix
                input matrix A

    @param
    b           magma_z_vector
                RHS b

    @param
    precond     magma_z_preconditioner*
                preconditioner parameter

    @ingroup magmasparse_zgesv
    ********************************************************************/

magma_int_t
magma_zpastixsetup( magma_z_sparse_matrix A, magma_z_vector b,
                        magma_z_preconditioner *precond ){

#if defined(HAVE_PASTIX)

    #if defined(PRECISION_d)

        pastix_data_t    *pastix_data = NULL; /* Pointer to a storage structure needed by pastix           */
        pastix_int_t      ncol;               /* Size of the matrix                                        */
        pastix_int_t     *colptr      = NULL; /* Indexes of first element of each column in row and values */
        pastix_int_t     *rows        = NULL; /* Row of each element of the matrix                         */
        pastix_float_t   *values      = NULL; /* Value of each element of the matrix                       */
        pastix_float_t   *rhs         = NULL; /* right hand side                                           */
        pastix_int_t     *iparm = NULL;  /* integer parameters for pastix                             */
        double           *dparm = NULL;  /* floating parameters for pastix                            */
        pastix_int_t     *perm        = NULL; /* Permutation tabular                                       */
        pastix_int_t     *invp        = NULL; /* Reverse permutation tabular                               */
        pastix_int_t      mat_type;

        magma_z_sparse_matrix A_h1, B;
        magma_z_vector diag, c_t, b_h;
        magma_z_vinit( &c_t, Magma_CPU, A.num_rows, MAGMA_Z_ZERO );
        magma_z_vinit( &diag, Magma_CPU, A.num_rows, MAGMA_Z_ZERO );
        magma_z_vtransfer( b, &b_h, A.memory_location, Magma_CPU);

        if( A.storage_type != Magma_CSR ){
            magma_z_mtransfer( A, &A_h1, A.memory_location, Magma_CPU);
            magma_z_mconvert( A_h1, &B, A_h1.storage_type, Magma_CSR);
        }
        else{
            magma_z_mtransfer( A, &B, A.memory_location, Magma_CPU);
        }


        rhs = (pastix_float_t*) b_h.val;
        ncol = B.num_rows;
        colptr = B.row;
        rows = B.col;
        values = (pastix_float_t*) B.val;

        mat_type = API_SYM_NO;

        iparm = (pastix_int_t*)malloc(IPARM_SIZE*sizeof(pastix_int_t));
        dparm = (pastix_float_t*)malloc(DPARM_SIZE*sizeof(pastix_float_t));

        /*******************************************/
        /* Initialize parameters to default values */
        /*******************************************/
        iparm[IPARM_MODIFY_PARAMETER]    = API_NO;
        pastix(&pastix_data, MPI_COMM_WORLD,
             ncol, colptr, rows, values,
             perm, invp, rhs, 1, iparm, dparm);
        iparm[IPARM_THREAD_NBR]          = 16;
        iparm[IPARM_SYM]                 = mat_type;
        iparm[IPARM_FACTORIZATION]       = API_FACT_LU;
        iparm[IPARM_VERBOSE]             = API_VERBOSE_YES;
        iparm[IPARM_ORDERING]            = API_ORDER_SCOTCH;
        iparm[IPARM_INCOMPLETE]          = API_NO;
        iparm[IPARM_RHS_MAKING]          = API_RHS_B;
        //iparm[IPARM_AMALGAMATION]         = 5;
        iparm[IPARM_LEVEL_OF_FILL]       = 0;
        /*  if (incomplete == 1)
            {
            dparm[DPARM_EPSILON_REFINEMENT] = 1e-7;
            }
        */


        /*
         * Matrix needs :
         *    - to be in fortran numbering
         *    - to have only the lower triangular part in symmetric case
         *    - to have a graph with a symmetric structure in unsymmetric case
         * If those criteria are not matched, the csc will be reallocated and changed. 
         */
        iparm[IPARM_MATRIX_VERIFICATION] = API_YES;

        perm = (pastix_int_t*)malloc(ncol*sizeof(pastix_int_t));
        invp = (pastix_int_t*)malloc(ncol*sizeof(pastix_int_t));

        /*******************************************/
        /*      Step 1 - Ordering / Scotch         */
        /*  Perform it only when the pattern of    */
        /*  matrix change.                         */
        /*  eg: mesh refinement                    */
        /*  In many cases users can simply go from */
        /*  API_TASK_ORDERING to API_TASK_ANALYSE  */
        /*  in one call.                           */
        /*******************************************/
        /*******************************************/
        /*      Step 2 - Symbolic factorization    */
        /*  Perform it only when the pattern of    */
        /*  matrix change.                         */
        /*******************************************/
        /*******************************************/
        /* Step 3 - Mapping and Compute scheduling */
        /*  Perform it only when the pattern of    */
        /*  matrix change.                         */
        /*******************************************/
        /*******************************************/
        /*     Step 4 - Numerical Factorisation    */
        /* Perform it each time the values of the  */
        /* matrix changed.                         */
        /*******************************************/

        iparm[IPARM_START_TASK] = API_TASK_ORDERING;
        iparm[IPARM_END_TASK]   = API_TASK_NUMFACT;

        pastix(&pastix_data, MPI_COMM_WORLD,
             ncol, colptr, rows, values,
             perm, invp, NULL, 1, iparm, dparm);

        precond->int_array_1 = (magma_int_t*) perm;
        precond->int_array_2 = (magma_int_t*) invp;

        precond->M.val = (magmaDoubleComplex*) values;
        precond->M.col = (magma_int_t*) colptr;
        precond->M.row = (magma_int_t*) rows;
        precond->M.num_rows = A.num_rows;
        precond->M.num_cols = A.num_cols;
        precond->M.memory_location = Magma_CPU;
        precond->pastix_data = pastix_data;
        precond->iparm = iparm;
        precond->dparm = dparm;

        if( A.storage_type != Magma_CSR){
            magma_z_mfree( &A_h1 );
        }   
        magma_z_vfree( &b_h);
        magma_z_mfree( &B );

    #else
        printf( "error: only double precision supported yet.\n");
    #endif

#else
        printf( "error: pastix not available.\n");
#endif

    return MAGMA_SUCCESS;

}








/**
    Purpose
    -------

    Applies the PASTIX LU
    
    Arguments
    ---------

    @param
    A           magma_z_sparse_matrix
                input matrix A

    @param
    b           magma_z_vector
                RHS b

    @param
    precond     magma_z_preconditioner*
                preconditioner parameter

    @ingroup magmasparse_
    ********************************************************************/

magma_int_t
magma_zapplypastix( magma_z_vector b, magma_z_vector *x, 
                    magma_z_preconditioner *precond ){

#if defined(HAVE_PASTIX)

    #if defined(PRECISION_d)

        pastix_int_t      ncol;               /* Size of the matrix                                        */
        pastix_int_t     *colptr      = NULL; /* Indexes of first element of each column in row and values */
        pastix_int_t     *rows        = NULL; /* Row of each element of the matrix                         */
        pastix_float_t   *values      = NULL; /* Value of each element of the matrix                       */
        pastix_float_t   *rhs         = NULL; /* right hand side                                           */
        pastix_int_t     *iparm;  /* integer parameters for pastix                             */
        double           *dparm;  /* floating parameters for pastix                            */
        pastix_int_t     *perm        = NULL; /* Permutation tabular                                       */
        pastix_int_t     *invp        = NULL; /* Reverse permutation tabular                               */

        magma_z_vector b_h;

        magma_z_vtransfer( b, &b_h, b.memory_location, Magma_CPU);

        rhs = (pastix_float_t*) b_h.val;
        ncol = precond->M.num_rows;
        colptr = (pastix_int_t*) precond->M.col;
        rows = (pastix_int_t*) precond->M.row;
        values = (pastix_float_t*) precond->M.val;
        iparm = precond->iparm;
        dparm = precond->dparm;

        perm = (pastix_int_t*)precond->int_array_1; 
        invp = (pastix_int_t*)precond->int_array_1; 

        /*******************************************/
        /*     Step 5 - Solve                      */
        /* For each one of your Right-hand-side    */
        /* members.                                */
        /* Also consider using multiple            */
        /* right-hand-side members.                */
        /*******************************************/
        iparm[IPARM_START_TASK] = API_TASK_SOLVE;
        iparm[IPARM_END_TASK]   = API_TASK_REFINEMENT;


        pastix(&(precond->pastix_data), MPI_COMM_WORLD,
             ncol, colptr, rows, values,
             perm, invp, b_h.val, 1, iparm, dparm);

        // fix that x is not allocated every time
        //  in case of many iterations, it might be faster to use
        // magma_zsetvector( ncol, 
        //                                    b_h.val, 1, x->val, 1 );
        magma_z_vfree( x );
        magma_z_vtransfer( b_h, x, Magma_CPU, b.memory_location);

        magma_z_vfree( &b_h);

    #else
        printf( "error: only double precision supported yet.\n");
    #endif

#else
        printf( "error: pastix not available.\n");
#endif

    return MAGMA_SUCCESS;

}
