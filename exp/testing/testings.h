#ifndef _TESTINGS_H_
#define _TESTINGS_H_

#ifndef min
#define min(a,b)  (((a)<(b))?(a):(b))
#endif
#ifndef max
#define max(a,b)  (((a)<(b))?(b):(a))
#endif

#define TESTING_CUDA_INIT()						\
  CUdevice  dev;							\
  CUcontext context;							\
  if( CUDA_SUCCESS != cuInit( 0 ) ) {					\
    fprintf(stderr, "CUDA: Not initialized\n" ); exit(-1);		\
  }									\
  if( CUDA_SUCCESS != cuDeviceGet( &dev, 0 ) ) {			\
    fprintf(stderr, "CUDA: Cannot get the device\n"); exit(-1);		\
  }									\
  if( CUDA_SUCCESS != cuCtxCreate( &context, 0, dev ) ) {		\
    fprintf(stderr, "CUDA: Cannot create the context\n"); exit(-1);	\
  }									\
  if( CUBLAS_STATUS_SUCCESS != cublasInit( ) ) {                        \
    fprintf(stderr, "CUBLAS: Not initialized\n"); exit(-1);		\
  }									\
  printout_devices( );


#define TESTING_CUDA_FINALIZE()			\
  cuCtxDetach( context );			\
  cublasShutdown();
     

#define TESTING_MALLOC(__ptr, __type, __size)                           \
  __ptr = (__type*)malloc((__size) * sizeof(__type));                   \
  if (__ptr == 0) {							\
    fprintf (stderr, "!!!! Malloc failed for: %s\n", #__ptr );		\
    exit(-1);								\
  }

#define TESTING_HOSTALLOC(__ptr, __type, __size)				\
  if( cudaSuccess != cudaMallocHost( (void**)&__ptr, (__size)*sizeof(__type) ) ) { \
    fprintf (stderr, "!!!! cudaMallocHost failed for: %s\n", #__ptr );	\
    exit(-1);								\
  }

#define TESTING_DEVALLOC(__ptr, __type, __size)				\
  if( cudaSuccess != cudaMalloc( (void**)&__ptr, (__size)*sizeof(__type) ) ){ \
    fprintf (stderr, "!!!! cublasAlloc failed for: %s\n", #__ptr );	\
    exit(-1);								\
  }


#define TESTING_FREE(__ptr)			\
  free(__ptr);

#define TESTING_HOSTFREE(__ptr)			\
  cudaFreeHost( __ptr );

#define TESTING_DEVFREE(__ptr)			\
  cudaFree( __ptr );

#endif /* _TESTINGS_H_ */
