/*
    -- MAGMA (version 1.6.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date January 2015

       @author Mark Gates
*/
#ifndef TRACE_H
#define TRACE_H

// has MagmaMaxGPUs, strlcpy, max
#include "common_magma.h"

// ----------------------------------------
const int MAX_CORES       = 1;                 // CPU cores
const int MAX_GPU_QUEUES  = MagmaMaxGPUs * 4;  // #devices * #queues per device
const int MAX_EVENTS      = 20000;
const int MAX_LABEL_LEN   = 16;


// ----------------------------------------
#ifdef TRACING

void trace_init     ( int ncore, int ngpu, int nqueue, magma_queue_t *queues );

void trace_cpu_start( int core, const char* tag, const char* label );
void trace_cpu_end  ( int core );

magma_event_t*
     trace_gpu_event( int dev, int queue_num, const char* tag, const char* label );
void trace_gpu_start( int dev, int queue_num, const char* tag, const char* label );
void trace_gpu_end  ( int dev, int queue_num );

void trace_finalize ( const char* filename, const char* cssfile );

#else

#define trace_init(      x1, x2, x3, x4 ) ((void)(0))

#define trace_cpu_start( x1, x2, x3     ) ((void)(0))
#define trace_cpu_end(   x1             ) ((void)(0))

#define trace_gpu_event( x1, x2, x3, x4 ) (NULL)
#define trace_gpu_start( x1, x2, x3, x4 ) ((void)(0))
#define trace_gpu_end(   x1, x2         ) ((void)(0))

#define trace_finalize(  x1, x2         ) ((void)(0))

#endif

#endif        //  #ifndef TRACE_H
