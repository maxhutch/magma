      SUBROUTINE  ZCHECK_EIG(JOBZ, MATYPE, N,NB,A,LDA, AD, AE, D1, EIG,
     $                     Z,LDZ,WORK,RWORK,RESU)
      IMPLICIT NONE
    
      CHARACTER          JOBZ
      INTEGER            MATYPE, N, NB, LDA, LDZ, OPTION
      DOUBLE PRECISION   D1( * ), EIG( * ), RESU(*),
     $                   AD(*), AE(*), RWORK(*)
      DOUBLE COMPLEX     A(LDA,*), WORK(*), Z(LDZ,*)

*     .. Local Scalars ..
      INTEGER            I, IINFO, IL, IMODE, ITEMP, ITYPE, IU, J, JC,
     $                   JR, JSIZE, JTYPE, LGN, LIWEDC, LOG2UI, LWEDC,
     $                   M, M2, M3, MTYPES, NAP, NBLOCK, NERRS,
     $                   NMATS, NMAX, NSPLIT, NTEST, NTESTT
      DOUBLE PRECISION   ABSTOL, ANINV, ANORM, COND, OVFL, RTOVFL,
     $                   RTUNFL, TEMP1, TEMP2, TEMP3, TEMP4, ULP,
     $                   ULPINV, UNFL, VL, VU

*     .. Parameters ..
      DOUBLE PRECISION   ZERO, MONE, ONE, TWO, EIGHT, TEN, HUN
      PARAMETER          ( ZERO = 0.0D0, MONE = -1.0D0, ONE = 1.0D0, 
     $                   TWO = 2.0D0, EIGHT = 8.0D0, TEN = 10.0D0, 
     $                   HUN = 100.0D0 )
      DOUBLE PRECISION   HALF
      PARAMETER          ( HALF = ONE / TWO )
      CHARACTER          UPLO
      DOUBLE PRECISION   DNOTHING, MAXABS_1, MAXABS_2, MAXNORM, MAXDOT
      DOUBLE COMPLEX     ZNOTHING

*
*     .. External Functions ..
      INTEGER            ILAENV, IDAMAX
      DOUBLE PRECISION   DLAMCH, DLARND, DSXT1, DNRM2, 
     $                   DLANGE, DLANSY, DDOT
      EXTERNAL           ILAENV, DLAMCH, DLARND, DSXT1
     $                   IDAMAX, DNRM2, DLANGE, DLANSY, DDOT
*     ..
*     .. External Subroutines ..
      EXTERNAL           DCOPY,  DLABAD, DLACPY, DLASET, DLASUM, DLATMR,
     $                   DLATMS, DOPGTR, DORGTR, DPTEQR, DSPT21, DSPTRD,
     $                   DSTEBZ, DSTECH, DSTEDC, DSTEMR, DSTEIN, DSTEQR,
     $                   DSTERF, DSTT21, DSTT22, DSYT21, DSYTRD, XERBLA,
     $                   DSBT21, DSCAL, ZHET21, ZHBT21, ZSTT21
*     ..
*     .. Intrinsic Functions ..
      INTRINSIC          CABS, ABS, DBLE, INT, LOG, MAX, MIN, SQRT
*     ..
*       the following tests will be performed:
*       (1)     | A - Z D Z' | / ( |A| n ulp )
*
*       (2)     | I - Z Z' | / ( n ulp )
*
*       (3)     | D1 - D2 | / ( |D1| ulp )
*
*
*     Quick return if possible
*
      IF( N.EQ.0 )
     $   RETURN
*
*     More Important constants
*
      UNFL = DLAMCH( 'Safe minimum' )
      OVFL = DLAMCH( 'Overflow' )
      CALL DLABAD( UNFL, OVFL )
      ULP = DLAMCH( 'Epsilon' )*DLAMCH( 'Base' )
      ULPINV = ONE / ULP
      LOG2UI = INT( LOG( ULPINV ) / LOG( TWO ) )
      RTUNFL = SQRT( UNFL )
      RTOVFL = SQRT( OVFL )
*
      RESU( 1 ) = ULPINV
      RESU( 2 ) = ULPINV
      RESU( 3 ) = ULPINV

*      PRINT*, 'VOICI ULP',ULP

*     ===============================
*     JOBZ = N check diag only
*     ===============================
      IF(JOBZ.EQ.'N') THEN
*        Do check (3)
         TEMP1 = ZERO
         TEMP2 = ZERO
         DO 210 J = 1, N
            TEMP1 = MAX( TEMP1, ABS( D1( J ) ), ABS( EIG( J ) ) )
            TEMP2 = MAX( TEMP2, ABS( D1( J ) - EIG( J ) ) )
 210     CONTINUE
         RESU( 3 ) = TEMP2 / MAX( UNFL,
     $                  ULP*MAX( TEMP1, TEMP2 ) )
         RESU( 7 ) = TEMP2 
*     ===============================
*     JOBZ = V  check all
*     ===============================
      ELSE
*        Do check (1) and (2)
*          
         IF(MATYPE.EQ.1)THEN
            CALL ZSTT21( N, 0, AD, AE, D1, DNOTHING, Z, LDZ, WORK,
     $                     RWORK, RESU( 1) )
         ELSEIF(MATYPE.EQ.2)THEN
            UPLO='L'
            CALL ZHBT21( UPLO, N, NB, 0, A, LDA, D1, DNOTHING, 
     $                  Z, LDZ, WORK, RWORK, RESU(1) )

         ELSEIF(MATYPE.EQ.3)THEN
            UPLO='L'
            CALL ZHET21( 1, UPLO, N, 0, A, LDA, D1, DNOTHING, Z, LDZ,
     $               ZNOTHING, LDZ, ZNOTHING, WORK, RWORK, RESU( 1 ) )
         ENDIF
    
*        Do check (3)
         TEMP1 = ZERO
         TEMP2 = ZERO
         DO 100 J = 1, N
            TEMP1 = MAX( TEMP1, ABS( D1( J ) ), ABS( EIG( J ) ) )
            TEMP2 = MAX( TEMP2, ABS( D1( J ) - EIG( J ) ) )
 100     CONTINUE
         RESU( 3 ) = TEMP2 / MAX( UNFL,
     $                  ULP*MAX( TEMP1, TEMP2 ) )
         RESU( 7 ) = TEMP2

C         WRITE(unit=6,FMT='(A,E15.5)') 
C     $   " | A - U S U' | / ( |A| n ulp )     ", RESU(1)
C         WRITE(unit=6,FMT='(A,E15.5)') 
C     $   " | I - U U' | / ( n ulp )           ", RESU(2) 
C         WRITE(unit=6,FMT='(A,E15.5)') 
C     $   " | D1 - EVEIGS | / (|D| ulp)        ", RESU(3)      


      

      ENDIF




 300  CONTINUE
      RETURN
      END

