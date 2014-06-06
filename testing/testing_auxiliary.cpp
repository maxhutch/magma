/*
    -- MAGMA (version 1.4.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       December 2013

       @author Mark Gates
*/
#include <stdlib.h>
#include <stdio.h>

// make sure that asserts are enabled
#ifdef NDEBUG
#undef NDEBUG
#endif
#include <assert.h>

#include "common_magma.h"

void test_num_gpus()
{
    int ngpu;
    int ndevices;
    cudaGetDeviceCount( &ndevices );
    int maxgpu = min( ndevices, MagmaMaxGPUs );
    
    printf( "$MAGMA_NUM_GPUS  ngpu\n" );
    printf( "=====================\n" );
    
#ifndef _MSC_VER // not Windows
    
    unsetenv("MAGMA_NUM_GPUS");
    ngpu = magma_num_gpus();
    printf( "%-15s  %d\n\n", "not set", ngpu );
    assert( ngpu == 1 );
    
    setenv("MAGMA_NUM_GPUS", "", 1 );
    ngpu = magma_num_gpus();
    printf( "%-15s  %d\n\n", getenv("MAGMA_NUM_GPUS"), ngpu );
    assert( ngpu == 1 );
    
    setenv("MAGMA_NUM_GPUS", "-1", 1 );
    ngpu = magma_num_gpus();
    printf( "%-15s  %d\n\n", getenv("MAGMA_NUM_GPUS"), ngpu );
    assert( ngpu == 1 );
    
    setenv("MAGMA_NUM_GPUS", "2junk", 1 );
    ngpu = magma_num_gpus();
    printf( "%-15s  %d\n\n", getenv("MAGMA_NUM_GPUS"), ngpu );
    assert( ngpu == 1 );
    
    setenv("MAGMA_NUM_GPUS", "1", 1 );
    ngpu = magma_num_gpus();
    printf( "%-15s  %d\n\n", getenv("MAGMA_NUM_GPUS"), ngpu );
    assert( ngpu == 1 );
    
    setenv("MAGMA_NUM_GPUS", "2", 1 );
    ngpu = magma_num_gpus();
    printf( "%-15s  %d\n\n", getenv("MAGMA_NUM_GPUS"), ngpu );
    assert( ngpu == min(  2, maxgpu ) );
    
    setenv("MAGMA_NUM_GPUS", "4", 1 );
    ngpu = magma_num_gpus();
    printf( "%-15s  %d\n\n", getenv("MAGMA_NUM_GPUS"), ngpu );
    assert( ngpu == min(  4, maxgpu ) );
    
    setenv("MAGMA_NUM_GPUS", "8", 1 );
    ngpu = magma_num_gpus();
    printf( "%-15s  %d\n\n", getenv("MAGMA_NUM_GPUS"), ngpu );
    assert( ngpu == min(  8, maxgpu ) );
    
    setenv("MAGMA_NUM_GPUS", "16", 1 );
    ngpu = magma_num_gpus();
    printf( "%-15s  %d\n\n", getenv("MAGMA_NUM_GPUS"), ngpu );
    assert( ngpu == min( 16, maxgpu ) );
    
    setenv("MAGMA_NUM_GPUS", "32", 1 );
    ngpu = magma_num_gpus();
    printf( "%-15s  %d\n\n", getenv("MAGMA_NUM_GPUS"), ngpu );
    assert( ngpu == min( 32, maxgpu ) );
    
#endif // not Windows
}

int main( int argc, char** argv )
{
    test_num_gpus();
    return 0;
}
