      SUBROUTINE DSCALDE(N,D,E,SIGMA,ISCALE,WORK,OPTION,INFO)
      IMPLICIT NONE

*     .. Scalar Arguments ..
      CHARACTER          UPLO
      INTEGER            INFO, LDA, N, ISCALE,OPTION

      DOUBLE PRECISION   SIGMA
      DOUBLE PRECISION   D( * ), E(*), WORK(*)


*  UPLO    (input) CHARACTER*1
*          = 'U':  Upper triangle of A is stored;
*          = 'L':  Lower triangle of A is stored.
*
*  N       (input) INTEGER
*          The order of the matrix A.  N >= 0.
*
*  A       (input/output) DOUBLE PRECISION array, dimension (LDA, N)
*          On entry, the symmetric matrix A.  If UPLO = 'U', the
*          leading N-by-N upper triangular part of A contains the
*          upper triangular part of the matrix A.  If UPLO = 'L',
*          the leading N-by-N lower triangular part of A contains
*          the lower triangular part of the matrix A.
*          On exit, if A require SCALING then a scaling has been done
*          with SIGMA.
*
*  LDA     (input) INTEGER
*          The leading dimension of the array A.  LDA >= max(1,N).
*      
*  SIGMA   (input/output) DOUBLE PRECISION
*          input: when option=2. this mean that scaling has been done
*          SIGMA contain the value of the scaling factor.      
*          output: when option=1. 
*          if scaling occur, SIGMA is the scaling factor that
*          should be used to scal back the matrix otherwise SIGMA =ONE
*      
*  ISCALE  (output) INTEGER
*          this integer indicate if scaling occur (=1) or not (=0) 
*      
*  OPTION (input) INTEGER
*          indicate whathever to do scale (1) or scale back (2)
*

*
*     .. Parameters ..
      DOUBLE PRECISION   ZERO, ONE, TWO
      PARAMETER          ( ZERO = 0.0D0, ONE = 1.0D0, TWO = 2.0D0 )
*     ..
*     .. Local Scalars ..
      LOGICAL            FULLMAT
      INTEGER            IINFO, IMAX, INDE, INDTAU, INDWRK, 
     $                   LLWORK, LWKOPT, NB
      DOUBLE PRECISION   ANRM, BIGNUM, EPS, RMAX, RMIN, SAFMIN, 
     $                   SMLNUM
*     ..
*     .. External Functions ..
      LOGICAL            LSAME
      INTEGER            ILAENV
      DOUBLE PRECISION   DLAMCH, DLANSY,DLANST
      EXTERNAL           LSAME, ILAENV, DLAMCH, DLANSY,DLANST
*     ..
*     .. External Subroutines ..
      EXTERNAL           DLASCL, DSCAL, XERBLA
*     ..
*     .. Intrinsic Functions ..
      INTRINSIC          ABS, DBLE, INT, LOG, MAX, MOD, SQRT


*
*     OPTION = 1 TRY TO DO SCALING
*      
      IF(OPTION.EQ.1)THEN 
*
*        Get machine constants.
*
         SAFMIN = DLAMCH( 'Safe minimum' )
         EPS = DLAMCH( 'Precision' )
         SMLNUM = SAFMIN / EPS
         BIGNUM = ONE / SMLNUM
         RMIN = SQRT( SMLNUM )
         RMAX = SQRT( BIGNUM )
    
*
*        Scale matrix to allowable range, if necessary.
*
         ISCALE=0
         SIGMA = DLANST( 'M', N, D, E )
         IF( SIGMA.EQ.ZERO )
     $      RETURN

*
*              Scale.
*
         ISCALE=1
               CALL DLASCL( 'G', 0, 0, SIGMA, ONE, N, 1, D, N,
     $                      INFO )
               CALL DLASCL( 'G', 0, 0, SIGMA, ONE, N-1, 1, E,
     $                      N-1, INFO )

*
*     OPTION = 2 IF SCALING HAPPEN ==> DO SCALING BACK
*    
      ELSEIF(OPTION.EQ.2)THEN
*
*        If matrix was scaled, then rescale eigenvalues appropriately.
*
         IF( ISCALE.EQ.1 ) THEN
               CALL DLASCL( 'G', 0, 0, ONE, SIGMA, N, 1, D, N,
     $                      INFO )
         END IF
         INFO=0
      ELSE 
          INFO =-1
      ENDIF




      RETURN
      END
