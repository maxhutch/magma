/*
    -- MAGMA (version 1.5.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date September 2014

       @precisions normal z -> s d c
       @author Hartwig Anzt

*/

#include <assert.h>

// includes CUDA
#include <cuda_runtime_api.h>
#include <cublas.h>
#include <cusparse_v2.h>
#include <cuda_profiler_api.h>


#include <stdio.h>  
#include <stdlib.h>
#include <sys/time.h>
#include <time.h>
#include <iostream>

/////////////////////////////////////
// CUDA imports 
/////////////////////////////////////
#include <cuda_runtime.h>
#include <cublas.h>

#include "magma_lapack.h"
#include "common_magma.h"
#include "magmasparse.h"

// includes, project
#include "magma.h"
#include "magmasparse.h"

#define BLOCK_SIZE 512
#define STREAM_COUNT 4

/////////////////////////////////////
// error checking 
/////////////////////////////////////
magma_int_t
magma_zcheckerr(const char *label)
{
    cudaThreadSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        const char *e = cudaGetErrorString(err);
        fprintf(stderr, "CUDA Error: %s (at %s)", e, label);
    }
    return MAGMA_SUCCESS; 
}


/**
    Purpose
    -------

    Initializes P2P communication between GPUs.

    Arguments
    ---------

    @param
    bw_bmark    magma_int_t*
                input: run the benchmark (1/0)

    @param
    num_gpus    magma_int_t*
                output: number of GPUs


    @ingroup magmasparse_zaux
    ********************************************************************/

magma_int_t
magma_z_initP2P ( magma_int_t *bw_bmark, magma_int_t *num_gpus ){


    // Number of GPUs
    printf("Checking for multiple GPUs...\n");
    int gpu_n;
     (cudaGetDeviceCount(&gpu_n));
    printf("CUDA-capable device count: %i\n", gpu_n);
    if (gpu_n < 2)
    {
        printf("Two or more Tesla(s) with (SM 2.0)"
                        " class GPUs are required for P2P.\n");
    }

    // Query device properties
    cudaDeviceProp prop[64];
    int gpuid_tesla[64]; // find the first two GPU's that can support P2P
    int gpu_count = 0;   // GPUs that meet the criteria

    for (int i=0; i < gpu_n; i++) {
         (cudaGetDeviceProperties(&prop[i], i));
        // Only Tesla boards based on Fermi can support P2P
        {
            // This is an array of P2P capable GPUs
            gpuid_tesla[gpu_count++] = i;
        }
    }
    *num_gpus=gpu_n;

     for(int i=0; i<gpu_n; i++)
    {
        for(int j=i+1; j<gpu_n; j++)
        {
  // Check possibility for peer access
    printf("\nChecking GPU(s) for support of peer to peer memory access...\n");
    int can_access_peer_0_1, can_access_peer_1_0;
    // In this case we just pick the first two that we can support
     (cudaDeviceCanAccessPeer(&can_access_peer_0_1, gpuid_tesla[i], 
                                                        gpuid_tesla[j]));
     (cudaDeviceCanAccessPeer(&can_access_peer_1_0, gpuid_tesla[j], 
                                                        gpuid_tesla[i]));


    // Output results from P2P capabilities
    printf("> Peer access from %s (GPU%d) -> %s (GPU%d) : %s\n", 
                        prop[gpuid_tesla[i]].name, gpuid_tesla[i],                                                                  
                        prop[gpuid_tesla[j]].name, gpuid_tesla[j] ,
                             can_access_peer_0_1 ? "Yes" : "No");
    printf("> Peer access from %s (GPU%d) -> %s (GPU%d) : %s\n", 
                        prop[gpuid_tesla[j]].name, gpuid_tesla[j],
                        prop[gpuid_tesla[i]].name, gpuid_tesla[i],
                            can_access_peer_1_0 ? "Yes" : "No");

    if (can_access_peer_0_1 == 0 || can_access_peer_1_0 == 0)
    {
        printf("Two or more Tesla(s) with class"
                " GPUs are required for P2P to run.\n");
        printf("Support for UVA requires a Tesla with SM 2.0 capabilities.\n");
        printf("Peer to Peer access is not available between"
        " GPU%d <-> GPU%d, waiving test.\n", gpuid_tesla[i], gpuid_tesla[j]);
        printf("PASSED\n");
        //exit(EXIT_SUCCESS);
    }
     }    
    }

  // Enable peer access
     for(int i=0; i<gpu_n; i++)
     {
         for(int j=i+1; j<gpu_n; j++)
         {
             printf("Enabling peer access between GPU%d and GPU%d...\n",
                gpuid_tesla[i], gpuid_tesla[j]);
              (cudaSetDevice(gpuid_tesla[i]));
              (cudaDeviceEnablePeerAccess(gpuid_tesla[j], 0));
              (cudaSetDevice(gpuid_tesla[j]));
              (cudaDeviceEnablePeerAccess(gpuid_tesla[i], 0));
           magma_zcheckerr("P2P");
         }
     }

   magma_zcheckerr("P2P successful");


    // Enable peer access
    for(int i=0; i<gpu_n; i++)
    {
        for(int j=i+1; j<gpu_n; j++)
        {
    // Check that we got UVA on both devices
    printf("Checking GPU%d and GPU%d for UVA capabilities...\n", 
    gpuid_tesla[i], gpuid_tesla[j]);
    //const bool has_uva = (prop[gpuid_tesla[i]].unifiedAddressing && 
    //                            prop[gpuid_tesla[j]].unifiedAddressing);

    printf("> %s (GPU%d) supports UVA: %s\n", prop[gpuid_tesla[i]].name, 
    gpuid_tesla[i], (prop[gpuid_tesla[i]].unifiedAddressing ? "Yes" : "No") );
    printf("> %s (GPU%d) supports UVA: %s\n", prop[gpuid_tesla[j]].name, 
    gpuid_tesla[j], (prop[gpuid_tesla[j]].unifiedAddressing ? "Yes" : "No") );

        }
    }



  if(*bw_bmark==1){


    // P2P memcopy() benchmark
   for(int i=0; i<gpu_n; i++)
    {
        for(int j=i+1; j<gpu_n; j++)
        {
    // Allocate buffers
    const size_t buf_size = 1024 * 1024 * 16 * sizeof(float);
    printf("Allocating buffers (%iMB on GPU%d, GPU%d and CPU Host)...\n", 
                int(buf_size / 1024 / 1024), gpuid_tesla[i], gpuid_tesla[j]);
    (cudaSetDevice(gpuid_tesla[i]));
    float* g0;
    (cudaMalloc(&g0, buf_size));
    (cudaSetDevice(gpuid_tesla[j]));
    float* g1;
    (cudaMalloc(&g1, buf_size));
    float* h0;
    (cudaMallocHost(&h0, buf_size)); // Automatically portable with UVA

    // Create CUDA event handles
    printf("Creating event handles...\n");
    cudaEvent_t start_event, stop_event;
    float time_memcpy;
    int eventflags = cudaEventBlockingSync;
    (cudaEventCreateWithFlags(&start_event, eventflags));
    (cudaEventCreateWithFlags(&stop_event, eventflags));


    (cudaEventRecord(start_event, 0));
    for (int k=0; k<100; k++)
    {
        // With UVA we don't need to specify source and target devices, the
        // runtime figures this out by itself from the pointers
            
        // Ping-pong copy between GPUs
        if (k % 2 == 0)
            (cudaMemcpy(g1, g0, buf_size, cudaMemcpyDefault));
        else
            (cudaMemcpy(g0, g1, buf_size, cudaMemcpyDefault));
    }
    (cudaEventRecord(stop_event, 0));
    (cudaEventSynchronize(stop_event));
    (cudaEventElapsedTime(&time_memcpy, start_event, stop_event));
    printf("cudaMemcpyPeer / cudaMemcpy between"
            "GPU%d and GPU%d: %.2fGB/s\n", gpuid_tesla[i], gpuid_tesla[j],
        (1.0f / (time_memcpy / 1000.0f)) 
            * ((100.0f * buf_size)) / 1024.0f / 1024.0f / 1024.0f);


     // Cleanup and shutdown
    printf("Cleanup of P2P benchmark...\n");
    (cudaEventDestroy(start_event));
    (cudaEventDestroy(stop_event));
    (cudaSetDevice(gpuid_tesla[i]));
    (magma_free( g0) );
    (cudaSetDevice(gpuid_tesla[j]));
    (magma_free( g1) );
    (magma_free_cpu( h0) );

    }
    }

    // host-device memcopy() benchmark

        for(int j=0; j<gpu_n; j++)
        {
    cudaSetDevice(gpuid_tesla[j]);

    int *h_data_source;
    int *h_data_sink;

    int *h_data_in[STREAM_COUNT];
    int *d_data_in[STREAM_COUNT];

    int *h_data_out[STREAM_COUNT];
    int *d_data_out[STREAM_COUNT];


    cudaEvent_t cycleDone[STREAM_COUNT];
    cudaStream_t stream[STREAM_COUNT];

    cudaEvent_t start, stop;

    // Allocate resources
    int memsize;
    memsize = 1000000 * sizeof(int);

    h_data_source = (int*) malloc(memsize);
    h_data_sink = (int*) malloc(memsize);    

    for( int i =0; i<STREAM_COUNT; ++i ) {
        
        ( cudaHostAlloc(&h_data_in[i], memsize, 
            cudaHostAllocDefault) );
        ( cudaMalloc(&d_data_in[i], memsize) );
        
        ( cudaHostAlloc(&h_data_out[i], memsize, 
            cudaHostAllocDefault) );
        ( cudaMalloc(&d_data_out[i], memsize) );

        
        ( cudaStreamCreate(&stream[i]) );
        ( cudaEventCreate(&cycleDone[i]) ); 
        
        cudaEventRecord(cycleDone[i], stream[i]);
    }

    cudaEventCreate(&start); cudaEventCreate(&stop);

    

    // Time host-device copies
    cudaEventRecord(start,0);
    ( cudaMemcpyAsync(d_data_in[0], h_data_in[0], memsize, 
        cudaMemcpyHostToDevice,0) );
    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);
    
    float memcpy_h2d_time;    
    cudaEventElapsedTime(&memcpy_h2d_time, start, stop);

    
    cudaEventRecord(start,0);
    ( cudaMemcpyAsync(h_data_out[0], d_data_out[0], memsize, 
        cudaMemcpyDeviceToHost, 0) );        
    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);
    
    float memcpy_d2h_time;    
    cudaEventElapsedTime(&memcpy_d2h_time, start, stop);
    
    cudaEventSynchronize(stop);
    printf("Measured timings (throughput):\n");
    printf(" Memcpy host to device GPU %d \t: %f ms (%f GB/s)\n", j,
        memcpy_h2d_time, (memsize * 1e-6)/ memcpy_h2d_time );
    printf(" Memcpy device GPU %d to host\t: %f ms (%f GB/s)\n", j,
        memcpy_d2h_time, (memsize * 1e-6)/ memcpy_d2h_time);

    // Free resources

    free( h_data_source );
    free( h_data_sink );

    for( int i =0; i<STREAM_COUNT; ++i ) {
        
        magma_free_cpu( h_data_in[i] );
        magma_free( d_data_in[i] );

        magma_free_cpu( h_data_out[i] );
        magma_free( d_data_out[i] );
        
        cudaStreamDestroy(stream[i]);
        cudaEventDestroy(cycleDone[i]);        
    }
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    
   }

  }//end if-loop bandwidth_benchmark

    magma_zcheckerr("P2P established");

    return MAGMA_SUCCESS;

}

