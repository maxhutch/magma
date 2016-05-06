/*
    -- MAGMA (version 2.0.2) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date May 2016

       @author Mark Gates
*/
#ifndef TRACE_H
#define TRACE_H

// has MagmaMaxGPUs, strlcpy, max
// TODO: what's the best way to protect inclusion?
#ifndef MAGMA_H
#include "magma_v2.h"
#endif

// ----------------------------------------
const magma_int_t MAX_CORES       = 1;                 // CPU cores
const magma_int_t MAX_GPU_QUEUES  = MagmaMaxGPUs * 4;  // #devices * #queues per device
const magma_int_t MAX_EVENTS      = 20000;
const magma_int_t MAX_LABEL_LEN   = 16;


// ----------------------------------------
#ifdef TRACING

void trace_init     ( magma_int_t ncore, magma_int_t ngpu, magma_int_t nqueue, magma_queue_t *queues );

void trace_cpu_start( magma_int_t core, const char* tag, const char* label );
void trace_cpu_end  ( magma_int_t core );

magma_event_t*
     trace_gpu_event( magma_int_t dev, magma_int_t queue_num, const char* tag, const char* label );
void trace_gpu_start( magma_int_t dev, magma_int_t queue_num, const char* tag, const char* label );
void trace_gpu_end  ( magma_int_t dev, magma_int_t queue_num );

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
