#ifndef TRACE_H
#define TRACE_H

// has MagmaMaxGPUs, strlcpy, max
#include "common_magma.h"

// ----------------------------------------
const int MAX_CORES       = 1;                 // CPU cores
const int MAX_GPU_STREAMS = MagmaMaxGPUs * 4;  // #devices * #streams per device
const int MAX_EVENTS      = 20000;
const int MAX_LABEL_LEN   = 16;


// ----------------------------------------
#ifdef TRACING

void trace_init     ( int nthreads, int ngpus, int nstream, cudaStream_t *streams );
void trace_cpu_start( int core, const char* tag, const char* label );
void trace_cpu_end  ( int core );
void trace_gpu_start( int core, int stream_num, const char* tag, const char* label );
void trace_gpu_end  ( int core, int stream_num );
void trace_finalize ( const char* filename, const char* cssfile );

#else 

#define trace_init(       x1, x2, x3, x4 ) ((void)(0))
#define trace_cpu_start(  x1, x2, x3     ) ((void)(0))
#define trace_cpu_end(    x1             ) ((void)(0))
#define trace_gpu_start(  x1, x2, x3, x4 ) ((void)(0))
#define trace_gpu_end(    x1, x2         ) ((void)(0))
#define trace_finalize(   x1, x2         ) ((void)(0))

#endif

#endif        //  #ifndef TRACE_H
