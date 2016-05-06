/*
    -- MAGMA (version 2.0.2) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date May 2016

       @generated from sparse-iter/blas/zmergeqmr.cu normal z -> d, Mon May  2 23:30:47 2016
       @author Hartwig Anzt

*/
#include "magmasparse_internal.h"

#define BLOCK_SIZE 512

#define PRECISION_d


// These routines merge multiple kernels from qmr into one.

/* -------------------------------------------------------------------------- */

__global__ void
magma_dqmr_1_kernel(  
    int num_rows, 
    int num_cols, 
    double rho,
    double psi,
    double *y, 
    double *z,
    double *v,
    double *w )
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if ( i<num_rows ) {
        for( int j=0; j<num_cols; j++ ){
            
            double ytmp = y[ i+j*num_rows ] / rho;
            y[ i+j*num_rows ] = ytmp;
            v[ i+j*num_rows ] = ytmp;
            
            double ztmp = z[ i+j*num_rows ] / psi;
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
    rho         double
                scalar
                
    @param[in]
    psi         double
                scalar
                
    @param[in,out]
    y           magmaDouble_ptr 
                vector
                
    @param[in,out]
    z           magmaDouble_ptr 
                vector
                
    @param[in,out]
    v           magmaDouble_ptr 
                vector

    @param[in,out]
    w           magmaDouble_ptr 
                vector

    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_dgegpuk
    ********************************************************************/

extern "C" 
magma_int_t
magma_dqmr_1(  
    magma_int_t num_rows, 
    magma_int_t num_cols, 
    double rho,
    double psi,
    magmaDouble_ptr y, 
    magmaDouble_ptr z,
    magmaDouble_ptr v,
    magmaDouble_ptr w,
    magma_queue_t queue )
{
    dim3 Bs( BLOCK_SIZE );
    dim3 Gs( magma_ceildiv( num_rows, BLOCK_SIZE ) );
    magma_dqmr_1_kernel<<< Gs, Bs, 0, queue->cuda_stream() >>>( num_rows, num_cols, rho, psi,
                     y, z, v, w );

   return MAGMA_SUCCESS;
}





__global__ void
magma_dqmr_2_kernel(  
    int num_rows,
    int num_cols,
    double pde,
    double rde,
    magmaDouble_ptr y,
    magmaDouble_ptr z,
    magmaDouble_ptr p, 
    magmaDouble_ptr q )
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
    pde         double
                scalar

    @param[in]
    qde         double
                scalar
                
    @param[in]
    y           magmaDouble_ptr 
                vector
                
    @param[in]
    z           magmaDouble_ptr 
                vector

    @param[in,out]
    p           magmaDouble_ptr 
                vector
                
    @param[in,out]
    q           magmaDouble_ptr 
                vector

    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_dgegpuk
    ********************************************************************/

extern "C" 
magma_int_t
magma_dqmr_2(  
    magma_int_t num_rows, 
    magma_int_t num_cols, 
    double pde,
    double rde,
    magmaDouble_ptr y,
    magmaDouble_ptr z,
    magmaDouble_ptr p, 
    magmaDouble_ptr q, 
    magma_queue_t queue )
{
    dim3 Bs( BLOCK_SIZE );
    dim3 Gs( magma_ceildiv( num_rows, BLOCK_SIZE ) );
    magma_dqmr_2_kernel<<< Gs, Bs, 0, queue->cuda_stream() >>>( num_rows, num_cols, pde, rde, y, z, p, q );

   return MAGMA_SUCCESS;
}





__global__ void
magma_dqmr_3_kernel(  
    int num_rows,
    int num_cols,
    double beta,
    double *pt,
    double *v,
    double *y )
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if ( i<num_rows ) {
        for( int j=0; j<num_cols; j++ ){
            double tmp = pt[ i+j*num_rows ] - beta * v[ i+j*num_rows ];
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
    beta        double
                scalar
                
    @param[in]
    pt          magmaDouble_ptr 
                vector

    @param[in,out]
    v           magmaDouble_ptr 
                vector
                
    @param[in,out]
    y           magmaDouble_ptr 
                vector

    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_dgegpuk
    ********************************************************************/

extern "C" 
magma_int_t
magma_dqmr_3(  
    magma_int_t num_rows, 
    magma_int_t num_cols, 
    double beta,
    magmaDouble_ptr pt,
    magmaDouble_ptr v,
    magmaDouble_ptr y,
    magma_queue_t queue )
{
    dim3 Bs( BLOCK_SIZE );
    dim3 Gs( magma_ceildiv( num_rows, BLOCK_SIZE ) );
    magma_dqmr_3_kernel<<< Gs, Bs, 0, queue->cuda_stream() >>>( num_rows, num_cols, beta, pt, v, y );

   return MAGMA_SUCCESS;
}




__global__ void
magma_dqmr_4_kernel(  
    int num_rows,
    int num_cols,
    double eta,
    double *p,
    double *pt,
    double *d,
    double *s,
    double *x,
    double *r )
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if ( i<num_rows ) {
        for( int j=0; j<num_cols; j++ ){
            
            double tmpd = eta * p[ i+j*num_rows ];
            d[ i+j*num_rows ] = tmpd;
            x[ i+j*num_rows ] = x[ i+j*num_rows ] + tmpd;
            double tmps = eta * pt[ i+j*num_rows ];
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
    eta         double
                scalar
                
    @param[in]
    p           magmaDouble_ptr 
                vector
                
    @param[in]
    pt          magmaDouble_ptr 
                vector

    @param[in,out]
    d           magmaDouble_ptr 
                vector
                
    @param[in,out]
    s           magmaDouble_ptr 
                vector
                
    @param[in,out]
    x           magmaDouble_ptr 
                vector
                
    @param[in,out]
    r           magmaDouble_ptr 
                vector

    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_dgegpuk
    ********************************************************************/

extern "C" 
magma_int_t
magma_dqmr_4(  
    magma_int_t num_rows, 
    magma_int_t num_cols, 
    double eta,
    magmaDouble_ptr p,
    magmaDouble_ptr pt,
    magmaDouble_ptr d, 
    magmaDouble_ptr s, 
    magmaDouble_ptr x, 
    magmaDouble_ptr r, 
    magma_queue_t queue )
{
    dim3 Bs( BLOCK_SIZE );
    dim3 Gs( magma_ceildiv( num_rows, BLOCK_SIZE ) );
    magma_dqmr_4_kernel<<< Gs, Bs, 0, queue->cuda_stream() >>>( num_rows, num_cols, eta, p, pt, d, s, x, r );

   return MAGMA_SUCCESS;
}


__global__ void
magma_dqmr_5_kernel(  
    int num_rows,
    int num_cols,
    double eta,
    double pds,
    double *p,
    double *pt,
    double *d,
    double *s,
    double *x,
    double *r )
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if ( i<num_rows ) {
        for( int j=0; j<num_cols; j++ ){
            
            double tmpd = eta * p[ i+j*num_rows ] + pds * d[ i+j*num_rows ];
            d[ i+j*num_rows ] = tmpd;
            x[ i+j*num_rows ] = x[ i+j*num_rows ] + tmpd;
            double tmps = eta * pt[ i+j*num_rows ] + pds * s[ i+j*num_rows ];
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
    eta         double
                scalar
                    
    @param[in]
    pds         double
                scalar
                
    @param[in]
    p           magmaDouble_ptr 
                vector
                
    @param[in]
    pt          magmaDouble_ptr 
                vector

    @param[in,out]
    d           magmaDouble_ptr 
                vector
                
    @param[in,out]
    s           magmaDouble_ptr 
                vector
                
    @param[in,out]
    x           magmaDouble_ptr 
                vector
                
    @param[in,out]
    r           magmaDouble_ptr 
                vector

    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_dgegpuk
    ********************************************************************/

extern "C" 
magma_int_t
magma_dqmr_5(  
    magma_int_t num_rows, 
    magma_int_t num_cols, 
    double eta,
    double pds,
    magmaDouble_ptr p,
    magmaDouble_ptr pt,
    magmaDouble_ptr d, 
    magmaDouble_ptr s, 
    magmaDouble_ptr x, 
    magmaDouble_ptr r, 
    magma_queue_t queue )
{
    dim3 Bs( BLOCK_SIZE );
    dim3 Gs( magma_ceildiv( num_rows, BLOCK_SIZE ) );
    magma_dqmr_5_kernel<<< Gs, Bs, 0, queue->cuda_stream() >>>( num_rows, num_cols, eta, pds, p, pt, d, s, x, r );

   return MAGMA_SUCCESS;
}


__global__ void
magma_dqmr_6_kernel(  
    int num_rows, 
    int num_cols, 
    double beta,
    double rho,
    double psi,
    double *y, 
    double *z,
    double *v,
    double *w,
    double *wt )
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if ( i<num_rows ) {
        for( int j=0; j<num_cols; j++ ){
            
            double wttmp = wt[ i+j*num_rows ]
                                - MAGMA_D_CONJ( beta ) * w[ i+j*num_rows ];
                                
            wt[ i+j*num_rows ] = wttmp;
            
            double ztmp = wttmp / psi;
            z[ i+j*num_rows ] = ztmp;
            w[ i+j*num_rows ] = ztmp;
            
            double ytmp = y[ i+j*num_rows ] / rho;
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
    beta        double
                scalar
                    
    @param[in]
    rho         double
                scalar
                
    @param[in]
    psi         double
                scalar
                
    @param[in,out]
    y           magmaDouble_ptr 
                vector
                
    @param[in,out]
    z           magmaDouble_ptr 
                vector
                
    @param[in,out]
    v           magmaDouble_ptr 
                vector

    @param[in,out]
    w           magmaDouble_ptr 
                vector
                    
    @param[in,out]
    wt          magmaDouble_ptr 
                vector

    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_dgegpuk
    ********************************************************************/

extern "C" 
magma_int_t
magma_dqmr_6(  
    magma_int_t num_rows, 
    magma_int_t num_cols, 
    double beta,
    double rho,
    double psi,
    magmaDouble_ptr y, 
    magmaDouble_ptr z,
    magmaDouble_ptr v,
    magmaDouble_ptr w,
    magmaDouble_ptr wt,
    magma_queue_t queue )
{
    dim3 Bs( BLOCK_SIZE );
    dim3 Gs( magma_ceildiv( num_rows, BLOCK_SIZE ) );
    magma_dqmr_6_kernel<<< Gs, Bs, 0, queue->cuda_stream() >>>( num_rows, num_cols, beta, rho, psi,
                     y, z, v, w, wt );

   return MAGMA_SUCCESS;
}
