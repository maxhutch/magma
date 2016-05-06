/*
    -- MAGMA (version 2.0.2) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date May 2016

       @generated from sparse-iter/blas/zmergeqmr.cu normal z -> c, Mon May  2 23:30:47 2016
       @author Hartwig Anzt

*/
#include "magmasparse_internal.h"

#define BLOCK_SIZE 512

#define PRECISION_c


// These routines merge multiple kernels from qmr into one.

/* -------------------------------------------------------------------------- */

__global__ void
magma_cqmr_1_kernel(  
    int num_rows, 
    int num_cols, 
    magmaFloatComplex rho,
    magmaFloatComplex psi,
    magmaFloatComplex *y, 
    magmaFloatComplex *z,
    magmaFloatComplex *v,
    magmaFloatComplex *w )
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if ( i<num_rows ) {
        for( int j=0; j<num_cols; j++ ){
            
            magmaFloatComplex ytmp = y[ i+j*num_rows ] / rho;
            y[ i+j*num_rows ] = ytmp;
            v[ i+j*num_rows ] = ytmp;
            
            magmaFloatComplex ztmp = z[ i+j*num_rows ] / psi;
            z[ i+j*num_rows ] = ztmp;
            w[ i+j*num_rows ] = ztmp;
            
        }
    }
}

/**
    Purpose
    -------

    Mergels multiple operations into one kernel:

    v = y / rho
    y = y / rho
    w = wt / psi
    z = z / psi
    
    @param[in]
    num_rows    magma_int_t
                dimension m
                
    @param[in]
    num_cols    magma_int_t
                dimension n
                
    @param[in]
    rho         magmaFloatComplex
                scalar
                
    @param[in]
    psi         magmaFloatComplex
                scalar
                
    @param[in,out]
    y           magmaFloatComplex_ptr 
                vector
                
    @param[in,out]
    z           magmaFloatComplex_ptr 
                vector
                
    @param[in,out]
    v           magmaFloatComplex_ptr 
                vector

    @param[in,out]
    w           magmaFloatComplex_ptr 
                vector

    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_cgegpuk
    ********************************************************************/

extern "C" 
magma_int_t
magma_cqmr_1(  
    magma_int_t num_rows, 
    magma_int_t num_cols, 
    magmaFloatComplex rho,
    magmaFloatComplex psi,
    magmaFloatComplex_ptr y, 
    magmaFloatComplex_ptr z,
    magmaFloatComplex_ptr v,
    magmaFloatComplex_ptr w,
    magma_queue_t queue )
{
    dim3 Bs( BLOCK_SIZE );
    dim3 Gs( magma_ceildiv( num_rows, BLOCK_SIZE ) );
    magma_cqmr_1_kernel<<< Gs, Bs, 0, queue->cuda_stream() >>>( num_rows, num_cols, rho, psi,
                     y, z, v, w );

   return MAGMA_SUCCESS;
}





__global__ void
magma_cqmr_2_kernel(  
    int num_rows,
    int num_cols,
    magmaFloatComplex pde,
    magmaFloatComplex rde,
    magmaFloatComplex_ptr y,
    magmaFloatComplex_ptr z,
    magmaFloatComplex_ptr p, 
    magmaFloatComplex_ptr q )
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if ( i<num_rows ) {
        for( int j=0; j<num_cols; j++ ){
            p[ i+j*num_rows ] = y[ i+j*num_rows ] - pde * p[ i+j*num_rows ];
            q[ i+j*num_rows ] = z[ i+j*num_rows ] - rde * q[ i+j*num_rows ];
        }
    }
}

/**
    Purpose
    -------

    Mergels multiple operations into one kernel:

    p = y - pde * p
    q = z - rde * q

    Arguments
    ---------

    @param[in]
    num_rows    magma_int_t
                dimension m
                
    @param[in]
    num_cols    magma_int_t
                dimension n
                
    @param[in]
    pde         magmaFloatComplex
                scalar

    @param[in]
    qde         magmaFloatComplex
                scalar
                
    @param[in]
    y           magmaFloatComplex_ptr 
                vector
                
    @param[in]
    z           magmaFloatComplex_ptr 
                vector

    @param[in,out]
    p           magmaFloatComplex_ptr 
                vector
                
    @param[in,out]
    q           magmaFloatComplex_ptr 
                vector

    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_cgegpuk
    ********************************************************************/

extern "C" 
magma_int_t
magma_cqmr_2(  
    magma_int_t num_rows, 
    magma_int_t num_cols, 
    magmaFloatComplex pde,
    magmaFloatComplex rde,
    magmaFloatComplex_ptr y,
    magmaFloatComplex_ptr z,
    magmaFloatComplex_ptr p, 
    magmaFloatComplex_ptr q, 
    magma_queue_t queue )
{
    dim3 Bs( BLOCK_SIZE );
    dim3 Gs( magma_ceildiv( num_rows, BLOCK_SIZE ) );
    magma_cqmr_2_kernel<<< Gs, Bs, 0, queue->cuda_stream() >>>( num_rows, num_cols, pde, rde, y, z, p, q );

   return MAGMA_SUCCESS;
}





__global__ void
magma_cqmr_3_kernel(  
    int num_rows,
    int num_cols,
    magmaFloatComplex beta,
    magmaFloatComplex *pt,
    magmaFloatComplex *v,
    magmaFloatComplex *y )
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if ( i<num_rows ) {
        for( int j=0; j<num_cols; j++ ){
            magmaFloatComplex tmp = pt[ i+j*num_rows ] - beta * v[ i+j*num_rows ];
            v[ i+j*num_rows ] = tmp;
            y[ i+j*num_rows ] = tmp;
        }
    }
}

/**
    Purpose
    -------

    Mergels multiple operations into one kernel:

    v = pt - beta * v
    y = v

    Arguments
    ---------

    @param[in]
    num_rows    magma_int_t
                dimension m
                
    @param[in]
    num_cols    magma_int_t
                dimension n
                
    @param[in]
    beta        magmaFloatComplex
                scalar
                
    @param[in]
    pt          magmaFloatComplex_ptr 
                vector

    @param[in,out]
    v           magmaFloatComplex_ptr 
                vector
                
    @param[in,out]
    y           magmaFloatComplex_ptr 
                vector

    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_cgegpuk
    ********************************************************************/

extern "C" 
magma_int_t
magma_cqmr_3(  
    magma_int_t num_rows, 
    magma_int_t num_cols, 
    magmaFloatComplex beta,
    magmaFloatComplex_ptr pt,
    magmaFloatComplex_ptr v,
    magmaFloatComplex_ptr y,
    magma_queue_t queue )
{
    dim3 Bs( BLOCK_SIZE );
    dim3 Gs( magma_ceildiv( num_rows, BLOCK_SIZE ) );
    magma_cqmr_3_kernel<<< Gs, Bs, 0, queue->cuda_stream() >>>( num_rows, num_cols, beta, pt, v, y );

   return MAGMA_SUCCESS;
}




__global__ void
magma_cqmr_4_kernel(  
    int num_rows,
    int num_cols,
    magmaFloatComplex eta,
    magmaFloatComplex *p,
    magmaFloatComplex *pt,
    magmaFloatComplex *d,
    magmaFloatComplex *s,
    magmaFloatComplex *x,
    magmaFloatComplex *r )
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if ( i<num_rows ) {
        for( int j=0; j<num_cols; j++ ){
            
            magmaFloatComplex tmpd = eta * p[ i+j*num_rows ];
            d[ i+j*num_rows ] = tmpd;
            x[ i+j*num_rows ] = x[ i+j*num_rows ] + tmpd;
            magmaFloatComplex tmps = eta * pt[ i+j*num_rows ];
            s[ i+j*num_rows ] = tmps;
            r[ i+j*num_rows ] = r[ i+j*num_rows ] - tmps;
        }
    }
}

/**
    Purpose
    -------

    Mergels multiple operations into one kernel:

    d = eta * p;
    s = eta * pt;
    x = x + d;
    r = r - s;

    Arguments
    ---------

    @param[in]
    num_rows    magma_int_t
                dimension m
                
    @param[in]
    num_cols    magma_int_t
                dimension n
                
    @param[in]
    eta         magmaFloatComplex
                scalar
                
    @param[in]
    p           magmaFloatComplex_ptr 
                vector
                
    @param[in]
    pt          magmaFloatComplex_ptr 
                vector

    @param[in,out]
    d           magmaFloatComplex_ptr 
                vector
                
    @param[in,out]
    s           magmaFloatComplex_ptr 
                vector
                
    @param[in,out]
    x           magmaFloatComplex_ptr 
                vector
                
    @param[in,out]
    r           magmaFloatComplex_ptr 
                vector

    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_cgegpuk
    ********************************************************************/

extern "C" 
magma_int_t
magma_cqmr_4(  
    magma_int_t num_rows, 
    magma_int_t num_cols, 
    magmaFloatComplex eta,
    magmaFloatComplex_ptr p,
    magmaFloatComplex_ptr pt,
    magmaFloatComplex_ptr d, 
    magmaFloatComplex_ptr s, 
    magmaFloatComplex_ptr x, 
    magmaFloatComplex_ptr r, 
    magma_queue_t queue )
{
    dim3 Bs( BLOCK_SIZE );
    dim3 Gs( magma_ceildiv( num_rows, BLOCK_SIZE ) );
    magma_cqmr_4_kernel<<< Gs, Bs, 0, queue->cuda_stream() >>>( num_rows, num_cols, eta, p, pt, d, s, x, r );

   return MAGMA_SUCCESS;
}


__global__ void
magma_cqmr_5_kernel(  
    int num_rows,
    int num_cols,
    magmaFloatComplex eta,
    magmaFloatComplex pds,
    magmaFloatComplex *p,
    magmaFloatComplex *pt,
    magmaFloatComplex *d,
    magmaFloatComplex *s,
    magmaFloatComplex *x,
    magmaFloatComplex *r )
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if ( i<num_rows ) {
        for( int j=0; j<num_cols; j++ ){
            
            magmaFloatComplex tmpd = eta * p[ i+j*num_rows ] + pds * d[ i+j*num_rows ];
            d[ i+j*num_rows ] = tmpd;
            x[ i+j*num_rows ] = x[ i+j*num_rows ] + tmpd;
            magmaFloatComplex tmps = eta * pt[ i+j*num_rows ] + pds * s[ i+j*num_rows ];
            s[ i+j*num_rows ] = tmps;
            r[ i+j*num_rows ] = r[ i+j*num_rows ] - tmps;
        }
    }
}

/**
    Purpose
    -------

    Mergels multiple operations into one kernel:

    d = eta * p + pds * d;
    s = eta * pt + pds * s;
    x = x + d;
    r = r - s;

    Arguments
    ---------

    @param[in]
    num_rows    magma_int_t
                dimension m
                
    @param[in]
    num_cols    magma_int_t
                dimension n
                
    @param[in]
    eta         magmaFloatComplex
                scalar
                    
    @param[in]
    pds         magmaFloatComplex
                scalar
                
    @param[in]
    p           magmaFloatComplex_ptr 
                vector
                
    @param[in]
    pt          magmaFloatComplex_ptr 
                vector

    @param[in,out]
    d           magmaFloatComplex_ptr 
                vector
                
    @param[in,out]
    s           magmaFloatComplex_ptr 
                vector
                
    @param[in,out]
    x           magmaFloatComplex_ptr 
                vector
                
    @param[in,out]
    r           magmaFloatComplex_ptr 
                vector

    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_cgegpuk
    ********************************************************************/

extern "C" 
magma_int_t
magma_cqmr_5(  
    magma_int_t num_rows, 
    magma_int_t num_cols, 
    magmaFloatComplex eta,
    magmaFloatComplex pds,
    magmaFloatComplex_ptr p,
    magmaFloatComplex_ptr pt,
    magmaFloatComplex_ptr d, 
    magmaFloatComplex_ptr s, 
    magmaFloatComplex_ptr x, 
    magmaFloatComplex_ptr r, 
    magma_queue_t queue )
{
    dim3 Bs( BLOCK_SIZE );
    dim3 Gs( magma_ceildiv( num_rows, BLOCK_SIZE ) );
    magma_cqmr_5_kernel<<< Gs, Bs, 0, queue->cuda_stream() >>>( num_rows, num_cols, eta, pds, p, pt, d, s, x, r );

   return MAGMA_SUCCESS;
}


__global__ void
magma_cqmr_6_kernel(  
    int num_rows, 
    int num_cols, 
    magmaFloatComplex beta,
    magmaFloatComplex rho,
    magmaFloatComplex psi,
    magmaFloatComplex *y, 
    magmaFloatComplex *z,
    magmaFloatComplex *v,
    magmaFloatComplex *w,
    magmaFloatComplex *wt )
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if ( i<num_rows ) {
        for( int j=0; j<num_cols; j++ ){
            
            magmaFloatComplex wttmp = wt[ i+j*num_rows ]
                                - MAGMA_C_CONJ( beta ) * w[ i+j*num_rows ];
                                
            wt[ i+j*num_rows ] = wttmp;
            
            magmaFloatComplex ztmp = wttmp / psi;
            z[ i+j*num_rows ] = ztmp;
            w[ i+j*num_rows ] = ztmp;
            
            magmaFloatComplex ytmp = y[ i+j*num_rows ] / rho;
            y[ i+j*num_rows ] = ytmp;
            v[ i+j*num_rows ] = ytmp;
            
        }
    }
}

/**
    Purpose
    -------

    Mergels multiple operations into one kernel:
    
    wt = wt - conj(beta) * w
    v = y / rho
    y = y / rho
    w = wt / psi
    z = wt / psi
    
    @param[in]
    num_rows    magma_int_t
                dimension m
                
    @param[in]
    num_cols    magma_int_t
                dimension n
                
    @param[in]
    beta        magmaFloatComplex
                scalar
                    
    @param[in]
    rho         magmaFloatComplex
                scalar
                
    @param[in]
    psi         magmaFloatComplex
                scalar
                
    @param[in,out]
    y           magmaFloatComplex_ptr 
                vector
                
    @param[in,out]
    z           magmaFloatComplex_ptr 
                vector
                
    @param[in,out]
    v           magmaFloatComplex_ptr 
                vector

    @param[in,out]
    w           magmaFloatComplex_ptr 
                vector
                    
    @param[in,out]
    wt          magmaFloatComplex_ptr 
                vector

    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_cgegpuk
    ********************************************************************/

extern "C" 
magma_int_t
magma_cqmr_6(  
    magma_int_t num_rows, 
    magma_int_t num_cols, 
    magmaFloatComplex beta,
    magmaFloatComplex rho,
    magmaFloatComplex psi,
    magmaFloatComplex_ptr y, 
    magmaFloatComplex_ptr z,
    magmaFloatComplex_ptr v,
    magmaFloatComplex_ptr w,
    magmaFloatComplex_ptr wt,
    magma_queue_t queue )
{
    dim3 Bs( BLOCK_SIZE );
    dim3 Gs( magma_ceildiv( num_rows, BLOCK_SIZE ) );
    magma_cqmr_6_kernel<<< Gs, Bs, 0, queue->cuda_stream() >>>( num_rows, num_cols, beta, rho, psi,
                     y, z, v, w, wt );

   return MAGMA_SUCCESS;
}
