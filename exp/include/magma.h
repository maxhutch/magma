/*
    -- MAGMA (version 1.6.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date January 2015
*/

#include <quark.h>

#ifndef _MAGMA_
#define _MAGMA_

/* ------------------------------------------------------------
 * MAGMA Blas Functions 
 * --------------------------------------------------------- */ 
#include "magmablas.h"

#include "auxiliary.h"

/* ------------------------------------------------------------
 * MAGMA Context
 * --------------------------------------------------------- */

typedef struct context
{
  /* Number of CPU core in this context */
  magma_int_t num_cores;

  /* Number of GPUs in this context */
  magma_int_t num_gpus;

  /* GPU contexts */
  CUcontext *gpu_context;

  /* QUARK scheduler */
  Quark *quark;

  /* Block size, internally used for some algorithms */
  magma_int_t nb;

  /* Pointer to other global algorithm-dependent parameters */
  void *params;

} magma_context;

/* ------------------------------------------------------------
 * MAGMA functions
 * --------------------------------------------------------- */
#include "magma_z.h"
#include "magma_c.h"
#include "magma_d.h"
#include "magma_s.h"
#include "magma_zc.h"
#include "magma_ds.h"

#define MagmaNoTrans       'N'
#define MagmaTrans         'T'
#define MagmaConjTrans     'C'

#define MagmaUpper         'U'
#define MagmaLower         'L'
#define MagmaUpperLower    'A'

#define MagmaNonUnit       'N'
#define MagmaUnit          'U'

#define MagmaLeft          'L'
#define MagmaRight         'R'

#define MagmaForward       'F'
#define MagmaBackward      'B'
                           
#define MagmaColumnwise    'C'
#define MagmaRowwise       'R'

#define MagmaNoVectors     'N'
#define MagmaVectors       'V'

#define MagmaNoTransStr    "NonTrans"
#define MagmaTransStr      "Trans"
#define MagmaConjTransStr  "Conj"

#define MagmaUpperStr      "Upper"
#define MagmaLowerStr      "Lower"
#define MagmaUpperLowerStr "All"

#define MagmaNonUnitStr    "NonUnit"
#define MagmaUnitStr       "Unit"

#define MagmaLeftStr       "Left"
#define MagmaRightStr      "Right"

#define MagmaForwardStr    "Forward"
#define MagmaBackwardStr   "Backward"

#define MagmaColumnwiseStr "Columnwise"
#define MagmaRowwiseStr    "Rowwise"

#define MagmaNoVectorsStr  "NoVectors"
#define MagmaVectorsStr    "Vectors"

/* ------------------------------------------------------------
 *   Return codes
 * --------------------------------------------------------- */
#define MAGMA_SUCCESS             0
#define MAGMA_ERR_ILLEGAL_VALUE  -4
#define MAGMA_ERR_ALLOCATION     -5
#define MAGMA_ERR_HOSTALLOC      -6
#define MAGMA_ERR_CUBLASALLOC    -7

/* ------------------------------------------------------------
 *   Macros to deal with cuda complex
 * --------------------------------------------------------- */
#define MAGMA_Z_SET2REAL(v, t)    (v).x = (t); (v).y = 0.0
#define MAGMA_Z_OP_NEG_ASGN(t, z) (t).x = -(z).x; (t).y = -(z).y
#define MAGMA_Z_EQUAL(u,v)        (((u).x == (v).x) && ((u).y == (v).y))
#define MAGMA_Z_GET_X(u)          ((u).x)
#define MAGMA_Z_ASSIGN(v, t)      (v).x = (t).x; (v).y = (t).y
#define MAGMA_Z_CNJG(v, t)        (v).x = (t).x; (v).y = -(t).y
#define MAGMA_Z_DSCALE(v, t, s)   (v).x = (t).x/(s); (v).y = (t).y/(s)      
#define MAGMA_Z_OP_NEG(a, b, c)   (a).x = (b).x-(c).x; (a).y = (b).y-(c).y
#define MAGMA_Z_MAKE(r, i)        make_cuDoubleComplex((r), (i))
#define MAGMA_Z_REAL(a)           cuCreal(a)
#define MAGMA_Z_IMAG(a)           cuCimag(a)
#define MAGMA_Z_ADD(a, b)         cuCadd((a), (b))
#define MAGMA_Z_SUB(a, b)         cuCsub((a), (b))
#define MAGMA_Z_MUL(a, b)         cuCmul((a), (b))
#define MAGMA_Z_DIV(a, b)         cuCdiv((a), (b))
#define MAGMA_Z_ABS(a)            cuCabs((a))
#define MAGMA_Z_ZERO              make_cuDoubleComplex(0.0, 0.0)
#define MAGMA_Z_ONE               make_cuDoubleComplex(1.0, 0.0)
#define MAGMA_Z_HALF              make_cuDoubleComplex(0.5, 0.0)
#define MAGMA_Z_NEG_ONE           make_cuDoubleComplex(-1.0, 0.0)
#define MAGMA_Z_NEG_HALF          make_cuDoubleComplex(-0.5, 0.0)

#define MAGMA_C_SET2REAL(v, t)    (v).x = (t); (v).y = 0.0
#define MAGMA_C_OP_NEG_ASGN(t, z) (t).x = -(z).x; (t).y = -(z).y
#define MAGMA_C_EQUAL(u,v)        (((u).x == (v).x) && ((u).y == (v).y))
#define MAGMA_C_GET_X(u)          ((u).x)
#define MAGMA_C_ASSIGN(v, t)      (v).x = (t).x; (v).y = (t).y
#define MAGMA_C_CNJG(v, t)        (v).x= (t).x; (v).y = -(t).y
#define MAGMA_C_DSCALE(v, t, s)   (v).x = (t).x/(s); (v).y = (t).y/(s)
#define MAGMA_C_OP_NEG(a, b, c)   (a).x = (b).x-(c).x; (a).y = (b).y-(c).y
#define MAGMA_C_MAKE(r, i)        make_cuFloatComplex((r), (i))
#define MAGMA_C_REAL(a)           cuCrealf(a)
#define MAGMA_C_IMAG(a)           cuCimagf(a)
#define MAGMA_C_ADD(a, b)         cuCaddf((a), (b))
#define MAGMA_C_SUB(a, b)         cuCsubf((a), (b))
#define MAGMA_C_MUL(a, b)         cuCmulf((a), (b))
#define MAGMA_C_DIV(a, b)         cuCdivf((a), (b))
#define MAGMA_C_ABS(a)            cuCabsf((a))
#define MAGMA_C_ZERO              make_cuFloatComplex(0.0, 0.0)
#define MAGMA_C_ONE               make_cuFloatComplex(1.0, 0.0)
#define MAGMA_C_HALF              make_cuFloatComplex(0.5, 0.0)
#define MAGMA_C_NEG_ONE           make_cuFloatComplex(-1.0, 0.0)
#define MAGMA_C_NEG_HALF          make_cuFloatComplex(-0.5, 0.0)

#define MAGMA_D_SET2REAL(v, t)    (v) = (t);
#define MAGMA_D_OP_NEG_ASGN(t, z) (t) = -(z)
#define MAGMA_D_EQUAL(u,v)        ((u) == (v))
#define MAGMA_D_GET_X(u)          (u)
#define MAGMA_D_ASSIGN(v, t)      (v) = (t)
#define MAGMA_D_CNJG(v, t)        (v) = (t)
#define MAGMA_D_DSCALE(v, t, s)   (v) = (t)/(s)
#define MAGMA_D_OP_NEG(a, b, c)   (a) = (b) - (c)
#define MAGMA_D_MAKE(r, i)        (r)
#define MAGMA_D_REAL(a)           (a)
#define MAGMA_D_IMAG(a)           (a)
#define MAGMA_D_ADD(a, b)         ( (a) + (b) )
#define MAGMA_D_SUB(a, b)         ( (a) - (b) )
#define MAGMA_D_MUL(a, b)         ( (a) * (b) )
#define MAGMA_D_DIV(a, b)         ( (a) / (b) )
#define MAGMA_D_ABS(a)            ((a)>0?(a):-(a))
#define MAGMA_D_ZERO              (0.0)
#define MAGMA_D_ONE               (1.0)
#define MAGMA_D_HALF              (0.5)
#define MAGMA_D_NEG_ONE           (-1.0)
#define MAGMA_D_NEG_HALF          (-0.5)

#define MAGMA_S_SET2REAL(v, t)    (v) = (t);
#define MAGMA_S_OP_NEG_ASGN(t, z) (t) = -(z)
#define MAGMA_S_EQUAL(u,v)        ((u) == (v))
#define MAGMA_S_GET_X(u)          (u)
#define MAGMA_S_ASSIGN(v, t)      (v) = (t)
#define MAGMA_S_CNJG(v, t)        (v) = (t)
#define MAGMA_S_DSCALE(v, t, s)   (v) = (t)/(s)
#define MAGMA_S_OP_NEG(a, b, c)   (a) = (b) - (c)
#define MAGMA_S_MAKE(r, i)        (r)
#define MAGMA_S_REAL(a)           (a)
#define MAGMA_S_IMAG(a)           (a)
#define MAGMA_S_ADD(a, b)         ( (a) + (b) )
#define MAGMA_S_SUB(a, b)         ( (a) - (b) )
#define MAGMA_S_MUL(a, b)         ( (a) * (b) )
#define MAGMA_S_DIV(a, b)         ( (a) / (b) )
#define MAGMA_S_ABS(a)            ((a)>0?(a):-(a))
#define MAGMA_S_ZERO              (0.0)
#define MAGMA_S_ONE               (1.0)
#define MAGMA_S_HALF              (0.5)
#define MAGMA_S_NEG_ONE           (-1.0)
#define MAGMA_S_NEG_HALF          (-0.5)

#ifndef CBLAS_SADDR
#define CBLAS_SADDR(a)  &(a)
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* ------------------------------------------------------------
 *   -- MAGMA function definitions
 * --------------------------------------------------------- */
void magma_xerbla( const char *name, magma_int_t info );
magma_context *magma_init(void *, void* (*func)(void *a), magma_int_t nthread, magma_int_t ncpu, 
                          magma_int_t ngpu, magma_int_t argc, char **argv);
void magma_finalize(magma_context *cntxt);
void auto_tune(char algorithm, char precision, magma_int_t ncores, magma_int_t ncorespsocket,
               magma_int_t m, magma_int_t n, magma_int_t *nb, magma_int_t *ob, magma_int_t *ib,
               magma_int_t *nthreads, magma_int_t *nquarkthreads);



#ifdef __cplusplus
}
#endif

#endif

