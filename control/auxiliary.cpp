/*
    -- MAGMA (version 1.6.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date January 2015
*/

#include "common_magma.h"
#include <assert.h>


// -------------------------
// Returns version of MAGMA, as defined by
// MAGMA_VERSION_MAJOR, MAGMA_VERSION_MINOR, MAGMA_VERSION_MICRO constants.
extern "C"
void magma_version( magma_int_t* major, magma_int_t* minor, magma_int_t* micro )
{
    if ( major != NULL && minor != NULL && micro != NULL ) {
        *major = MAGMA_VERSION_MAJOR;
        *minor = MAGMA_VERSION_MINOR;
        *micro = MAGMA_VERSION_MICRO;
    }
}


/**
    Purpose
    -------
    For debugging purposes, determines whether a pointer points to CPU or GPU memory.
    
    On CUDA architecture 2.0 cards with unified addressing, CUDA can tell if
    it is a device pointer or pinned host pointer.
    For malloc'd host pointers, cudaPointerGetAttributes returns error,
    implying it is a (non-pinned) host pointer.
    
    On older cards, this cannot determine if it is CPU or GPU memory.
    
    @param A   pointer to test
    
    @return
      -         1:  if A is a device pointer (definitely),
      -         0:  if A is a host   pointer (definitely or inferred from error),
      -        -1:  if unknown.

    @author Mark Gates
    @ingroup magma_util
    ********************************************************************/
extern "C"
magma_int_t magma_is_devptr( const void* A )
{
    cudaError_t err;
    cudaDeviceProp prop;
    cudaPointerAttributes attr;
    int dev;  // must be int
    err = cudaGetDevice( &dev );
    if ( ! err ) {
        err = cudaGetDeviceProperties( &prop, dev );
        if ( ! err && prop.unifiedAddressing ) {
            // I think the cudaPointerGetAttributes prototype is wrong, missing const (mgates)
            err = cudaPointerGetAttributes( &attr, const_cast<void*>( A ));
            if ( ! err ) {
                // definitely know type
                return (attr.memoryType == cudaMemoryTypeDevice);
            }
            else if ( err == cudaErrorInvalidValue ) {
                // clear error; see http://icl.cs.utk.edu/magma/forum/viewtopic.php?f=2&t=529
                cudaGetLastError();
                // infer as host pointer
                return 0;
            }
        }
    }
    // clear error
    cudaGetLastError();
    // unknown, e.g., device doesn't support unified addressing
    return -1;
}


/* ////////////////////////////////////////////////////////////////////////////
   -- Get number of GPUs to use from $MAGMA_NUM_GPUS environment variable.
   @author Mark Gates
*/
extern "C"
magma_int_t magma_num_gpus( void )
{
    const char *ngpu_str = getenv("MAGMA_NUM_GPUS");
    magma_int_t ngpu = 1;
    if ( ngpu_str != NULL ) {
        char* endptr;
        ngpu = strtol( ngpu_str, &endptr, 10 );
        int ndevices;  // must be int
        cudaGetDeviceCount( &ndevices );
        // if *endptr == '\0' then entire string was valid number (or empty)
        if ( ngpu < 1 || *endptr != '\0' ) {
            ngpu = 1;
            fprintf( stderr, "$MAGMA_NUM_GPUS='%s' is an invalid number; using %d GPU.\n",
                     ngpu_str, (int) ngpu );
        }
        else if ( ngpu > MagmaMaxGPUs || ngpu > ndevices ) {
            ngpu = min( ndevices, MagmaMaxGPUs );
            fprintf( stderr, "$MAGMA_NUM_GPUS='%s' exceeds MagmaMaxGPUs=%d or available GPUs=%d; using %d GPUs.\n",
                     ngpu_str, MagmaMaxGPUs, ndevices, (int) ngpu );
        }
        assert( 1 <= ngpu && ngpu <= ndevices );
    }
    return ngpu;
}


/* ////////////////////////////////////////////////////////////////////////////
   -- Auxiliary function: ipiv(i) indicates that row i has been swapped with
      ipiv(i) from top to bottom. This function rearranges ipiv into newipiv
      where row i has to be moved to newipiv(i). The new pivoting allows for
      parallel processing vs the original one assumes a specific ordering and
      has to be done sequentially.
*/
extern "C"
void swp2pswp( magma_trans_t trans, magma_int_t n, magma_int_t *ipiv, magma_int_t *newipiv)
{
  magma_int_t i, newind, ind;
  magma_int_t    notran = (trans == MagmaNoTrans);

  for(i=0; i<n; i++)
    newipiv[i] = -1;
  
  if (notran){
    for(i=0; i<n; i++){
      newind = ipiv[i] - 1;
      if (newipiv[newind] == -1) {
        if (newipiv[i]==-1){
          newipiv[i] = newind;
          if (newind>i)
            newipiv[newind]= i;
        }
        else
          {
            ind = newipiv[i];
            newipiv[i] = newind;
            if (newind>i)
              newipiv[newind]= ind;
          }
      }
      else {
        if (newipiv[i]==-1){
          if (newind>i){
            ind = newipiv[newind];
            newipiv[newind] = i;
            newipiv[i] = ind;
          }
          else
            newipiv[i] = newipiv[newind];
        }
        else{
          ind = newipiv[i];
          newipiv[i] = newipiv[newind];
          if (newind > i)
            newipiv[newind] = ind;
        }
      }
    }
  } else {
    for(i=n-1; i>=0; i--){
      newind = ipiv[i] - 1;
      if (newipiv[newind] == -1) {
        if (newipiv[i]==-1){
          newipiv[i] = newind;
          if (newind>i)
            newipiv[newind]= i;
        }
        else
          {
            ind = newipiv[i];
            newipiv[i] = newind;
            if (newind>i)
              newipiv[newind]= ind;
          }
      }
      else {
        if (newipiv[i]==-1){
          if (newind>i){
            ind = newipiv[newind];
            newipiv[newind] = i;
            newipiv[i] = ind;
          }
          else
            newipiv[i] = newipiv[newind];
        }
        else{
          ind = newipiv[i];
          newipiv[i] = newipiv[newind];
          if (newind > i)
            newipiv[newind] = ind;
        }
      }
    }
  }
}

// --------------------
// Convert global indices [j0, j1) to local indices [dj0, dj1) on GPU dev,
// according to 1D block cyclic distribution.
// Note j0 and dj0 are inclusive, while j1 and dj1 are exclusive.
// This is consistent with the C++ container notion of first and last.
//
// Example with n = 75, nb = 10, ngpu = 3
// local dj:                0- 9, 10-19, 20-29
// -------------------------------------------
// gpu 0: 2  blocks, cols:  0- 9, 30-39, 60-69
// gpu 1: 1+ blocks, cols: 10-19, 40-49, 70-74 (partial)
// gpu 2: 1  block , cols: 20-29, 50-59
//
// j0 = 15, j0dev = 1
// j1 = 70, j1dev = 0
// gpu 0: dj0 = 10, dj1 = 30
// gpu 1: dj0 =  5, dj1 = 20
// gpu 2: dj0 =  0, dj1 = 20
//
// @author Mark Gates

extern "C"
void magma_indices_1D_bcyclic( magma_int_t nb, magma_int_t ngpu, magma_int_t dev,
                               magma_int_t j0, magma_int_t j1,
                               magma_int_t* dj0, magma_int_t* dj1 )
{
    // on GPU jdev, which contains j0, dj0 maps to j0.
    // on other GPUs, dj0 is start of the block on that GPU after j0's block.
    magma_int_t jblock = (j0 / nb) / ngpu;
    magma_int_t jdev   = (j0 / nb) % ngpu;
    if ( dev < jdev ) {
        jblock += 1;
    }
    *dj0 = jblock*nb;
    if ( dev == jdev ) {
        *dj0 += (j0 % nb);
    }
    
    // on GPU jdev, which contains j1-1, dj1 maps to j1.
    // on other GPUs, dj1 is end of the block on that GPU before j1's block.
    // j1 points to element after end (e.g., n), so subtract 1 to get last
    // element, compute index, then add 1 to get just after that index again.
    j1 -= 1;
    jblock = (j1 / nb) / ngpu;
    jdev   = (j1 / nb) % ngpu;
    if ( dev > jdev ) {
        jblock -= 1;
    }
    if ( dev == jdev ) {
        *dj1 = jblock*nb + (j1 % nb) + 1;
    }
    else {
        *dj1 = jblock*nb + nb;
    }
}
