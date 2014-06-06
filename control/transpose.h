/*
    -- MAGMA (version 1.5.0-beta2) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date May 2014
 
       @author Mathieu Faverge
 
       Macro to transpose matrices before and after computation
       in LU kernels
*/

#ifndef MAGMA_TRANSPOSE_H
#define MAGMA_TRANSPOSE_H

#define magmablas_sgetmo_in( dA, dAT, ldda, m, n )              \
  dAT = dA;                                                     \
  if ( ( (m) == (n) ) && ( (m)%32 == 0) && ( (ldda)%32 == 0) ){ \
    magmablas_stranspose_inplace( ldda, dAT, ldda );            \
  } else {                                                      \
    if (MAGMA_SUCCESS != magma_smalloc(&dAT, (m)*(n)) )		\
      return MAGMA_ERR_DEVICE_ALLOC;                            \
    magmablas_stranspose2( dAT, ldda, dA, ldda, m, n );         \
  }

#define magmablas_sgetmo_out( dA, dAT, ldda, m, n )             \
  if ( ( (m) == (n) ) && ( (m)%32 == 0) && ( (ldda)%32 == 0) ){ \
    magmablas_stranspose_inplace( ldda, dAT, ldda );            \
  } else {                                                      \
    magmablas_stranspose2( dA, ldda, dAT, ldda, n, m );         \
    magma_free(dAT);                                            \
  }

#define magmablas_dgetmo_in( dA, dAT, ldda, m, n )              \
  dAT = dA;                                                     \
  if ( ( (m) == (n) ) && ( (m)%32 == 0) && ( (ldda)%32 == 0) ){ \
    magmablas_dtranspose_inplace( ldda, dAT, ldda );            \
  } else {                                                      \
    if (MAGMA_SUCCESS != magma_dmalloc(&dAT, (m)*(n)))		\
      return MAGMA_ERR_DEVICE_ALLOC;                            \
    magmablas_dtranspose2( dAT, ldda, dA, ldda, m, n );         \
  }

#define magmablas_dgetmo_out( dA, dAT, ldda, m, n )             \
  if ( ( (m) == (n) ) && ( (m)%32 == 0) && ( (ldda)%32 == 0) ){ \
    magmablas_dtranspose_inplace( ldda, dAT, ldda );            \
  } else {                                                      \
    magmablas_dtranspose2( dA, ldda, dAT, ldda, n, m );         \
    magma_free(dAT);                                            \
  }

#define magmablas_cgetmo_in( dA, dAT, ldda, m, n )              \
  dAT = dA;                                                     \
  if ( ( (m) == (n) ) && ( (m)%32 == 0) && ( (ldda)%32 == 0) ){ \
    magmablas_ctranspose_inplace( ldda, dAT, ldda );            \
  } else {                                                      \
    if (MAGMA_SUCCESS != magma_cmalloc(&dAT, (m)*(n)))		\
      return MAGMA_ERR_DEVICE_ALLOC;                            \
    magmablas_ctranspose2( dAT, ldda, dA, ldda, m, n );         \
  }

#define magmablas_cgetmo_out( dA, dAT, ldda, m, n )             \
  if ( ( (m) == (n) ) && ( (m)%32 == 0) && ( (ldda)%32 == 0) ){ \
    magmablas_ctranspose_inplace( ldda, dAT, ldda );            \
  } else {                                                      \
    magmablas_ctranspose2( dA, ldda, dAT, ldda, n, m );         \
    magma_free(dAT);                                            \
  }

#define magmablas_zgetmo_in( dA, dAT, ldda, m, n )              \
  dAT = dA;                                                     \
  if ( ( (m) == (n) ) && ( (m)%32 == 0) && ( (ldda)%32 == 0) ){ \
    magmablas_ztranspose_inplace( ldda, dAT, ldda );            \
  } else {                                                      \
    if (MAGMA_SUCCESS != magma_zmalloc(&dAT, (m)*(n)))		\
      return MAGMA_ERR_DEVICE_ALLOC;                            \
    magmablas_ztranspose2( dAT, ldda, dA, ldda, m, n );         \
  }

#define magmablas_zgetmo_out( dA, dAT, ldda, m, n )             \
  if ( ( (m) == (n) ) && ( (m)%32 == 0) && ( (ldda)%32 == 0) ){ \
    magmablas_ztranspose_inplace( ldda, dAT, ldda );            \
  } else {                                                      \
    magmablas_ztranspose2( dA, ldda, dAT, ldda, n, m );         \
    magma_free(dAT);                                            \
  }

#endif /* MAGMA_TRANSPOSE_H */
