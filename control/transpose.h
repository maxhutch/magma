/*
    -- MAGMA (version 1.6.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date January 2015
 
       @author Mathieu Faverge
 
       Macro to transpose matrices before and after computation
       in LU kernels
*/

#ifndef MAGMA_TRANSPOSE_H
#define MAGMA_TRANSPOSE_H

// TODO shouldn't need % 32 == 0 checks anymore; inplace works for any m == n.

#define magmablas_sgetmo_in( dA, dAT, ldda, m, n )              \
  dAT = dA;                                                     \
  if ( ( (m) == (n) ) && ( (m)%32 == 0) && ( (ldda)%32 == 0) ){ \
    magmablas_stranspose_inplace( ldda, dAT, ldda );            \
  } else {                                                      \
    if (MAGMA_SUCCESS != magma_smalloc(&dAT, (m)*(n)) )         \
      return MAGMA_ERR_DEVICE_ALLOC;                            \
    magmablas_stranspose( m, n, dA, ldda, dAT, ldda );          \
  }

#define magmablas_sgetmo_out( dA, dAT, ldda, m, n )             \
  if ( ( (m) == (n) ) && ( (m)%32 == 0) && ( (ldda)%32 == 0) ){ \
    magmablas_stranspose_inplace( ldda, dAT, ldda );            \
  } else {                                                      \
    magmablas_stranspose( n, m, dAT, ldda, dA, ldda );          \
    magma_free(dAT);                                            \
  }

#define magmablas_dgetmo_in( dA, dAT, ldda, m, n )              \
  dAT = dA;                                                     \
  if ( ( (m) == (n) ) && ( (m)%32 == 0) && ( (ldda)%32 == 0) ){ \
    magmablas_dtranspose_inplace( ldda, dAT, ldda );            \
  } else {                                                      \
    if (MAGMA_SUCCESS != magma_dmalloc(&dAT, (m)*(n)))          \
      return MAGMA_ERR_DEVICE_ALLOC;                            \
    magmablas_dtranspose( m, n, dA, ldda, dAT, ldda );          \
  }

#define magmablas_dgetmo_out( dA, dAT, ldda, m, n )             \
  if ( ( (m) == (n) ) && ( (m)%32 == 0) && ( (ldda)%32 == 0) ){ \
    magmablas_dtranspose_inplace( ldda, dAT, ldda );            \
  } else {                                                      \
    magmablas_dtranspose( n, m, dAT, ldda, dA, ldda );          \
    magma_free(dAT);                                            \
  }

#define magmablas_cgetmo_in( dA, dAT, ldda, m, n )              \
  dAT = dA;                                                     \
  if ( ( (m) == (n) ) && ( (m)%32 == 0) && ( (ldda)%32 == 0) ){ \
    magmablas_ctranspose_inplace( ldda, dAT, ldda );            \
  } else {                                                      \
    if (MAGMA_SUCCESS != magma_cmalloc(&dAT, (m)*(n)))          \
      return MAGMA_ERR_DEVICE_ALLOC;                            \
    magmablas_ctranspose( m, n, dA, ldda, dAT, ldda );          \
  }

#define magmablas_cgetmo_out( dA, dAT, ldda, m, n )             \
  if ( ( (m) == (n) ) && ( (m)%32 == 0) && ( (ldda)%32 == 0) ){ \
    magmablas_ctranspose_inplace( ldda, dAT, ldda );            \
  } else {                                                      \
    magmablas_ctranspose( n, m, dAT, ldda, dA, ldda );          \
    magma_free(dAT);                                            \
  }

#define magmablas_zgetmo_in( dA, dAT, ldda, m, n )              \
  dAT = dA;                                                     \
  if ( ( (m) == (n) ) && ( (m)%32 == 0) && ( (ldda)%32 == 0) ){ \
    magmablas_ztranspose_inplace( ldda, dAT, ldda );            \
  } else {                                                      \
    if (MAGMA_SUCCESS != magma_zmalloc(&dAT, (m)*(n)))          \
      return MAGMA_ERR_DEVICE_ALLOC;                            \
    magmablas_ztranspose( m, n, dA, ldda, dAT, ldda );          \
  }

#define magmablas_zgetmo_out( dA, dAT, ldda, m, n )             \
  if ( ( (m) == (n) ) && ( (m)%32 == 0) && ( (ldda)%32 == 0) ){ \
    magmablas_ztranspose_inplace( ldda, dAT, ldda );            \
  } else {                                                      \
    magmablas_ztranspose( n, m, dAT, ldda, dA, ldda );          \
    magma_free(dAT);                                            \
  }

#endif /* MAGMA_TRANSPOSE_H */
