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

#define RTOLERANCE     10e-16
#define ATOLERANCE     10e-16

#define  blockinfo(i,j)  A.blockinfo[(i)*c_blocks   + (j)]
#define M(i,j) M->val+((blockinfo(i,j)-1)*size_b*size_b)
#define A(i,j) A.val+((blockinfo(i,j)-1)*size_b*size_b)
#define x(i) x->val+(i*size_b)

/**
    Purpose
    -------

    LU for a BCSR matrix A. 
    We assume all diagonal blocks to be nonzero.

    Arguments
    ---------

    @param
    A           magma_z_sparse_matrix
                input matrix A (on DEV)

    @param
    M           magma_z_sparse_matrix*
                output matrix M approx. (LU)^{-1} (on DEV)

    @param
    ipiv        magma_int_t* 
                pivoting vector

    @ingroup magmasparse_zgesv
    ********************************************************************/

magma_int_t
magma_zilusetup( magma_z_sparse_matrix A, magma_z_sparse_matrix *M, magma_int_t *ipiv ){


    // some useful variables
    magmaDoubleComplex c_zero = MAGMA_Z_ZERO, c_one = MAGMA_Z_ONE, c_mone = MAGMA_Z_NEG_ONE;

    magma_int_t i,j,k,l,info;

    magma_int_t stat;


    // fill in information for B
    M->storage_type = A.storage_type;
    M->memory_location = Magma_DEV;
    M->num_rows = A.num_rows;
    M->num_cols = A.num_cols;
    M->nnz = A.nnz;
    M->max_nnz_row = A.max_nnz_row;
    M->blocksize = A.blocksize;
    M->numblocks = A.numblocks;
    magma_int_t size_b = A.blocksize;
    magma_int_t c_blocks = ceil( (float)A.num_cols / (float)size_b );     // max number of blocks per row
    magma_int_t r_blocks = ceil( (float)A.num_rows / (float)size_b );     // max number of blocks per column


    // memory allocation
    stat = magma_zmalloc( &M->val, size_b*size_b*A.numblocks );
    if( stat != 0 ) {printf("Memory Allocation Error transferring matrix\n"); exit(0); }
    stat = magma_imalloc( &M->row, r_blocks + 1  );
    if( stat != 0 ) {printf("Memory Allocation Error transferring matrix\n"); exit(0); } 
    stat = magma_imalloc( &M->col, A.numblocks );
    if( stat != 0 ) {printf("Memory Allocation Error transferring matrix\n"); exit(0); }
    magma_imalloc_cpu( &M->blockinfo, r_blocks * c_blocks );
    // data transfer
    magma_zcopyvector( size_b*size_b*A.numblocks, A.val, 1, M->val, 1 );
    magma_icopyvector( (r_blocks+1), A.row, 1, M->row, 1 );
    magma_icopyvector( A.numblocks, A.col, 1, M->col, 1 );
    for( magma_int_t i=0; i<r_blocks * c_blocks; i++ ){
        M->blockinfo[i] = A.blockinfo[i];
    }


    magma_index_t *cpu_row, *cpu_col;

    magma_imalloc_cpu( &cpu_row, r_blocks+1 );
    magma_imalloc_cpu( &cpu_col, A.numblocks );
    magma_igetvector( r_blocks+1, A.row, 1, cpu_row, 1 );            
    magma_igetvector( A.numblocks, A.col, 1, cpu_col, 1 );


    magma_int_t ldda, lddb, lddc, ldwork, lwork;
    magmaDoubleComplex tmp;
    // magma_imalloc_cpu( &ipiv, size_b*(r_blocks+1) );

    ldda = size_b;//((size_b+31)/32)*32;
    lddb = size_b;//((size_b+31)/32)*32;
    lddc = size_b;//((size_b+31)/32)*32;



    magmaDoubleComplex *inverse;
    magma_zmalloc( &inverse, ldda*size_b );

    magmaDoubleComplex *identity, *zerom, *identity_cpu;
    magma_zmalloc_cpu( &identity_cpu, size_b*size_b );
    for( i=0; i<size_b*size_b; i++ )
        identity_cpu[i] = c_zero;

    magma_zmalloc( &zerom, size_b*ldda );
    magma_zsetvector( size_b*size_b, identity_cpu, 1, zerom, 1 ); 

    for( i=0; i<size_b; i++ )
        identity_cpu[i*(size_b+1)] = c_one;

    magma_zmalloc( &identity, size_b*ldda );
    magma_zsetvector( size_b*size_b, identity_cpu, 1, identity, 1 ); 


    magmaDoubleComplex *dwork;
    ldwork = size_b * magma_get_zgetri_nb( size_b );
    magma_zmalloc( &dwork, ldwork );

    /*
    magmaDoubleComplex *work;
    // query for lapack workspace size 
    lwork = -1;
    lapackf77_zgetri( &size_b, inverse, &size_b, ipiv, &tmp, &lwork, &info ); // do we have to modify? take the max?
    if (info != 0)
        printf("lapackf77_zgetri returned error %d: %s.\n",
               (int) info, magma_strerror( info ));
    lwork = int( MAGMA_Z_REAL( tmp ));
    magma_zmalloc_cpu( &work, lwork );
    */

    // kij-version
    for( k=0; k<r_blocks; k++){

        magma_zgetrf_gpu( size_b, size_b, M(k,k), ldda, ipiv+k*size_b, &info );

        for( j=k+1; j<c_blocks; j++ ){
            if( (blockinfo(k,j)!=0) ){
                // Swap elements on the right before update
                magmablas_zlaswpx(size_b, M(k,j), 1, size_b,
                          1, size_b, ipiv+k*size_b, 1);
                // update
                magma_ztrsm(MagmaLeft, MagmaLower, MagmaNoTrans, MagmaUnit,
                size_b, size_b, c_one,
                M(k,k), ldda, M(k,j), size_b );
            }
        }
        // ------ in one routine (merged blocks)------
    /*
        magma_int_t count = 0;
        for( j=k+1; j<c_blocks; j++ ){
            if( (blockinfo(k,j)!=0) ){
                   count++;
            }
        }
           // Swap elements on the right before update
        magmablas_zlaswpx(size_b*count, M(k,j), 1, size_b*count,
                  1, size_b, ipiv+k*size_b, 1);
        // update          
        magma_ztrsm(MagmaLeft, MagmaLower, MagmaNoTrans, MagmaUnit,
        size_b, size_b, c_one,
        M(k,k), ldda, M(k,j), size_b );
    */
        // ------- in one routine --------------------

        // Swap elements on the left
        for( j=0; j<k; j++ ){
            if( (blockinfo(k,j)!=0) ){
                magmablas_zlaswpx(size_b, M(k,j), 1, size_b,
                  1, size_b,
                  ipiv+k*size_b, 1);                  
            }
        }

        for( i=k+1; i<c_blocks; i++ ){
            if( (blockinfo(i,k)!=0) && (i!=k) ){
                // update blocks below
                magma_ztrsm(MagmaRight, MagmaUpper, MagmaNoTrans, MagmaNonUnit,
                    size_b, size_b, c_one,
                    M(k,k), ldda, M(i,k), size_b);
                // update the blocks in the respective rows
                for( j=k+1; j<c_blocks; j++ ){
                    if( (blockinfo(i,j)!=0) && (blockinfo(k,j)!=0) ){
                        magmablas_zgemm( MagmaNoTrans, MagmaNoTrans, size_b, size_b, size_b,
                                         c_mone, M(i,k), size_b,
                                         M(k,j), size_b, c_one,  M(i,j), size_b );
                    }
                }
            }
        }
    }
    
    //magma_z_mvisu( *M );


    magma_free(inverse);
    magma_free(identity); 
    magma_free(zerom); 
    

    //free(ipiv); 

    free(identity_cpu);
    free(cpu_row);
    free(cpu_col);

    return MAGMA_SUCCESS;
}   /* magma_zilusetup */






magma_int_t
magma_zilu( magma_z_sparse_matrix A, magma_z_vector b, magma_z_vector *x,  
           magma_z_solver_par *solver_par, magma_int_t *ipiv ){

    // some useful variables
    magmaDoubleComplex one = MAGMA_Z_MAKE(1.0, 0.0);
    magmaDoubleComplex mone = MAGMA_Z_MAKE(-1.0, 0.0);
    magmaDoubleComplex zero = MAGMA_Z_MAKE(0.0, 0.0);
    magma_int_t i,j,k;
    magma_int_t size_b = A.blocksize;
    magma_int_t c_blocks = ceil( (float)A.num_cols / (float)size_b );     // max number of blocks per row
    magma_int_t r_blocks = ceil( (float)A.num_rows / (float)size_b );     // max number of blocks per column

    // set x = b
    magma_zcopyvector( A.num_rows, b.val, 1, x->val, 1 );


    // now solve
    /*
    // First pivot the RHS
    for( k=0; k<r_blocks; k++){
      magmablas_zlaswpx(1, x(k), 1, size_b,
            1, size_b,
            ipiv+k*size_b, 1);
    }
    */
    magmaDoubleComplex *work; 
    magma_zmalloc_cpu( &work, r_blocks*size_b );

    int nrhs = 1, n = r_blocks*size_b, ione = 1, inc = 1;
    magma_zgetmatrix(n, 1, x(0), n, work, n);
    for( k=0; k<r_blocks; k++)
      lapackf77_zlaswp(&nrhs, work+k*size_b, &n, &ione, &size_b, ipiv+k*size_b, &inc);
    magma_zsetmatrix(n, 1, work, n, x(0), n);
   
    magma_free_cpu(work);

    // forward solve
    for( k=0; k<r_blocks; k++){

        // do the forward triangular solve for block M(k,k): L(k,k)y = b
        magma_ztrsv(MagmaLower, MagmaNoTrans, MagmaUnit, size_b, A(k,k), size_b, x(k), 1 );

         // update for all nonzero blocks below M(k,k) the respective values of y
        for( j=k+1; j<c_blocks; j++ ){
            if( (blockinfo(j,k)!=0) ){
                //
                magmablas_zgemv( MagmaNoTrans, size_b, size_b, 
                                 mone, A(j,k), size_b,
                                 x(k), 1, one,  x(j), 1 );

            }
        }
    }

    // backward solve
    for( k=r_blocks-1; k>=0; k--){
        // do the backward triangular solve for block M(k,k): L(k,k)y = b
        magma_ztrsv(MagmaUpper, MagmaNoTrans, MagmaNonUnit, size_b, A(k,k), size_b, x(k), 1 );

        // update for all nonzero blocks above M(k,k) the respective values of y
        for( j=k-1; j>=0; j-- ){
            if( (blockinfo(j,k)!=0) ){
                magmablas_zgemv( MagmaNoTrans, size_b, size_b, 
                                 mone, A(j,k), size_b,
                                 x(k), 1, one,  x(j), 1 );

            }
        }
    }


    return MAGMA_SUCCESS;
}
