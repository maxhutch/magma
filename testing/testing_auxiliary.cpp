/*
    -- MAGMA (version 1.6.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date January 2015

       @author Mark Gates
*/
#include <stdlib.h>
#include <stdio.h>

// tests internal routines: magma_{set,get}_lapack_numthreads, magma_get_parallel_numthreads
// so include common_magma.h instead of magma.h
#include "common_magma.h"


////////////////////////////////////////////////////////////////////////////
// warn( condition ) is like assert, but doesn't abort. Also counts number of failures.
int gFailures = 0;

void warn_helper( int cond, const char* str, const char* file, int line )
{
    if ( ! cond ) {
        printf( "WARNING: %s:%d: assertion %s failed\n", file, line, str );
        gFailures += 1;
    }
}

#define warn(x) warn_helper( (x), #x, __FILE__, __LINE__ )


////////////////////////////////////////////////////////////////////////////
void test_num_gpus()
{
    printf( "======================================================================\n%s\n", __func__ );
    
    int ngpu;
    int ndevices;
    cudaGetDeviceCount( &ndevices );
    int maxgpu = min( ndevices, MagmaMaxGPUs );
    
    printf( "$MAGMA_NUM_GPUS     ngpu     expect\n" );
    printf( "===================================\n" );
    
#ifndef _MSC_VER // not Windows
    
    unsetenv("MAGMA_NUM_GPUS");
    ngpu = magma_num_gpus();
    printf( "%-18s  %7d  %6d\n\n", "not set", ngpu, 1 );
    warn( ngpu == 1 );
    
    setenv("MAGMA_NUM_GPUS", "", 1 );
    ngpu = magma_num_gpus();
    printf( "%-18s  %7d  %6d\n\n", getenv("MAGMA_NUM_GPUS"), ngpu, 1 );
    warn( ngpu == 1 );
    
    setenv("MAGMA_NUM_GPUS", "-1", 1 );
    ngpu = magma_num_gpus();
    printf( "%-18s  %7d  %6d\n\n", getenv("MAGMA_NUM_GPUS"), ngpu, 1 );
    warn( ngpu == 1 );
    
    setenv("MAGMA_NUM_GPUS", "2junk", 1 );
    ngpu = magma_num_gpus();
    printf( "%-18s  %7d  %6d\n\n", getenv("MAGMA_NUM_GPUS"), ngpu, 1 );
    warn( ngpu == 1 );
    
    setenv("MAGMA_NUM_GPUS", "0", 1 );
    ngpu = magma_num_gpus();
    printf( "%-18s  %7d  %6d\n\n", getenv("MAGMA_NUM_GPUS"), ngpu, 1 );
    warn( ngpu == 1 );
    
    setenv("MAGMA_NUM_GPUS", "1", 1 );
    ngpu = magma_num_gpus();
    printf( "%-18s  %7d  %6d\n\n", getenv("MAGMA_NUM_GPUS"), ngpu, 1 );
    warn( ngpu == 1 );
    
    setenv("MAGMA_NUM_GPUS", "2", 1 );
    ngpu = magma_num_gpus();
    printf( "%-18s  %7d  %6d\n\n", getenv("MAGMA_NUM_GPUS"), ngpu, 2 );
    warn( ngpu == min(  2, maxgpu ) );
    
    setenv("MAGMA_NUM_GPUS", "4", 1 );
    ngpu = magma_num_gpus();
    printf( "%-18s  %7d  %6d\n\n", getenv("MAGMA_NUM_GPUS"), ngpu, 4 );
    warn( ngpu == min(  4, maxgpu ) );
    
    setenv("MAGMA_NUM_GPUS", "8", 1 );
    ngpu = magma_num_gpus();
    printf( "%-18s  %7d  %6d\n\n", getenv("MAGMA_NUM_GPUS"), ngpu, 8 );
    warn( ngpu == min(  8, maxgpu ) );
    
    setenv("MAGMA_NUM_GPUS", "16", 1 );
    ngpu = magma_num_gpus();
    printf( "%-18s  %7d  %6d\n\n", getenv("MAGMA_NUM_GPUS"), ngpu, 16 );
    warn( ngpu == min( 16, maxgpu ) );
    
    setenv("MAGMA_NUM_GPUS", "1000", 1 );
    ngpu = magma_num_gpus();
    printf( "%-18s  %7d  %6d (maxgpu)\n\n", getenv("MAGMA_NUM_GPUS"), ngpu, maxgpu );
    warn( ngpu == min( 1000, maxgpu ) );
    
#endif // not Windows
}


////////////////////////////////////////////////////////////////////////////
void test_num_threads()
{
    printf( "======================================================================\n%s\n", __func__ );
    
    // test that getting & setting numthreads works
    int p_nthread_orig = magma_get_parallel_numthreads();
    int l_nthread_orig = magma_get_lapack_numthreads();
    printf( "get;      parallel_numthread=%2d, lapack_numthread=%2d\n",
            p_nthread_orig, l_nthread_orig );
    
    magma_set_lapack_numthreads( 4 );
    int p_nthread = magma_get_parallel_numthreads();
    int l_nthread = magma_get_lapack_numthreads();
    printf( "set( 4);  parallel_numthread=%2d, lapack_numthread=%2d (expect  4)\n",
            p_nthread, l_nthread );
    warn( p_nthread == p_nthread_orig );
    warn( l_nthread == 4 );
    
    magma_set_lapack_numthreads( 1 );
    p_nthread = magma_get_parallel_numthreads();
    l_nthread = magma_get_lapack_numthreads();
    printf( "set( 1);  parallel_numthread=%2d, lapack_numthread=%2d (expect  1)\n",
            p_nthread, l_nthread );
    warn( p_nthread == p_nthread_orig );
    warn( l_nthread == 1 );
    
    magma_set_lapack_numthreads( 8 );
    p_nthread = magma_get_parallel_numthreads();
    l_nthread = magma_get_lapack_numthreads();
    printf( "set( 8);  parallel_numthread=%2d, lapack_numthread=%2d (expect  8)\n",
            p_nthread, l_nthread );
    warn( p_nthread == p_nthread_orig );
    warn( l_nthread == 8 );
    
    magma_set_lapack_numthreads( l_nthread_orig );
    p_nthread = magma_get_parallel_numthreads();
    l_nthread = magma_get_lapack_numthreads();
    printf( "set(%2d);  parallel_numthread=%2d, lapack_numthread=%2d (expect %2d)\n",
            l_nthread_orig, p_nthread, l_nthread, l_nthread_orig );
    warn( p_nthread == p_nthread_orig );
    warn( l_nthread == l_nthread_orig );
    
#ifndef _MSC_VER // not Windows
    // test that parsing MAGMA_NUM_THREADS works
    
    // TODO need some way to get ncores. This is circular: assume with huge
    // NUM_THREADS that the routine gives the ncores. The user can verify.
    setenv("MAGMA_NUM_THREADS", "10000", 1 );
    int ncores = magma_get_parallel_numthreads();
    
    int omp_threads = ncores;
    const char* omp_str = getenv("OMP_NUM_THREADS");
    if ( omp_str != NULL ) {
        omp_threads = atoi( omp_str );
    }
    
    printf( "\nusing ncores=%d, omp_num_threads=%d\n\n", ncores, omp_threads );
    
    printf( "$MAGMA_NUM_THREADS  nthread  expect\n" );
    printf( "===================================\n" );
    
    unsetenv("MAGMA_NUM_THREADS");
    p_nthread = magma_get_parallel_numthreads();
    printf( "%-18s  %7d  %6d (omp_threads)\n\n", "not set", p_nthread, omp_threads );
    warn( p_nthread == omp_threads );
    
    setenv("MAGMA_NUM_THREADS", "", 1 );
    p_nthread = magma_get_parallel_numthreads();
    printf( "%-18s  %7d  %6d\n\n", getenv("MAGMA_NUM_THREADS"), p_nthread, 1 );
    warn( p_nthread == 1 );
    
    setenv("MAGMA_NUM_THREADS", "-1", 1 );
    p_nthread = magma_get_parallel_numthreads();
    printf( "%-18s  %7d  %6d\n\n", getenv("MAGMA_NUM_THREADS"), p_nthread, 1 );
    warn( p_nthread == 1 );
    
    setenv("MAGMA_NUM_THREADS", "2junk", 1 );
    p_nthread = magma_get_parallel_numthreads();
    printf( "%-18s  %7d  %6d\n\n", getenv("MAGMA_NUM_THREADS"), p_nthread, 1 );
    warn( p_nthread == 1 );
    
    setenv("MAGMA_NUM_THREADS", "0", 1 );
    p_nthread = magma_get_parallel_numthreads();
    printf( "%-18s  %7d  %6d\n\n", getenv("MAGMA_NUM_THREADS"), p_nthread, 1 );
    warn( p_nthread == 1 );
    
    setenv("MAGMA_NUM_THREADS", "1", 1 );
    p_nthread = magma_get_parallel_numthreads();
    printf( "%-18s  %7d  %6d\n\n", getenv("MAGMA_NUM_THREADS"), p_nthread, 1 );
    warn( p_nthread == 1 );
    
    setenv("MAGMA_NUM_THREADS", "2", 1 );
    p_nthread = magma_get_parallel_numthreads();
    printf( "%-18s  %7d  %6d\n\n", getenv("MAGMA_NUM_THREADS"), p_nthread, 2 );
    warn( p_nthread == min(  2, ncores ) );
    
    setenv("MAGMA_NUM_THREADS", "4", 1 );
    p_nthread = magma_get_parallel_numthreads();
    printf( "%-18s  %7d  %6d\n\n", getenv("MAGMA_NUM_THREADS"), p_nthread, 4 );
    warn( p_nthread == min(  4, ncores ) );
    
    setenv("MAGMA_NUM_THREADS", "8", 1 );
    p_nthread = magma_get_parallel_numthreads();
    printf( "%-18s  %7d  %6d\n\n", getenv("MAGMA_NUM_THREADS"), p_nthread, 8 );
    warn( p_nthread == min(  8, ncores ) );
    
    setenv("MAGMA_NUM_THREADS", "16", 1 );
    p_nthread = magma_get_parallel_numthreads();
    printf( "%-18s  %7d  %6d\n\n", getenv("MAGMA_NUM_THREADS"), p_nthread, 16 );
    warn( p_nthread == min( 16, ncores ) );
    
    setenv("MAGMA_NUM_THREADS", "1000", 1 );
    p_nthread = magma_get_parallel_numthreads();
    printf( "%-18s  %7d  %6d (ncores)\n\n", getenv("MAGMA_NUM_THREADS"), p_nthread, ncores );
    warn( p_nthread == min( 1000, ncores ) );
#endif // not Windows
}


////////////////////////////////////////////////////////////////////////////
void test_xerbla()
{
    magma_int_t info;
    info = -MAGMA_ERR_DEVICE_ALLOC;  magma_xerbla( __func__, -(info) );
    info = -MAGMA_ERR_HOST_ALLOC;    magma_xerbla( __func__, -(info) );
    info = -MAGMA_ERR;               magma_xerbla( __func__, -(info) );
    info =  3;                       magma_xerbla( __func__, -(info) );
    info =  2;                       magma_xerbla( __func__, -(info) );
    info =  1;                       magma_xerbla( __func__, -(info) );
    info =  0;                       magma_xerbla( __func__, -(info) );
    info = -1;                       magma_xerbla( __func__, -(info) );
    info = -2;                       magma_xerbla( __func__, -(info) );
    info = -3;                       magma_xerbla( __func__, -(info) );
    info = MAGMA_ERR;                magma_xerbla( __func__, -(info) );
    info = MAGMA_ERR_HOST_ALLOC;     magma_xerbla( __func__, -(info) );
    info = MAGMA_ERR_DEVICE_ALLOC;   magma_xerbla( __func__, -(info) );
}


////////////////////////////////////////////////////////////////////////////
int main( int argc, char** argv )
{
    test_num_gpus();
    test_num_threads();
    test_xerbla();
    
    if ( gFailures > 0 ) {
        printf( "\n%d tests failed.\n", gFailures );
    }
    else {
        printf( "\nAll tests passed.\n" );
    }
    
    return 0;
}
