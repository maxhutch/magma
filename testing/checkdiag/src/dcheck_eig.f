      SUBROUTINE  DCHECK_EIG(JOBZ, MATYPE, N,NB,A,LDA, AD, AE, D1, EIG,
     $                     Z,LDZ,WORK,RWORK,RESU)
      IMPLICIT NONE
    
      CHARACTER          JOBZ
      INTEGER            MATYPE, N, NB, LDA, LDZ, OPTION
      DOUBLE PRECISION   D1( * ), EIG( * ), RESU(*),
     $                   A(LDA,*), WORK(*), Z(LDZ,*),
     $                   AD(*), AE(*), RWORK(*)
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
      DOUBLE PRECISION   NOTHING, MAXABS_1, MAXABS_2, MAXNORM, MAXDOT

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
     $                   DSBT21, DSCAL
*     ..
*     .. Intrinsic Functions ..
      INTRINSIC          ABS, DBLE, INT, LOG, MAX, MIN, SQRT
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
            CALL DSTT21( N, 0, AD, AE, D1, NOTHING, Z, LDZ, WORK,
     $                      RESU( 1) )
         ELSEIF(MATYPE.EQ.2)THEN
            UPLO='L'
            CALL DSBT21( UPLO, N, NB, 0, A, LDA, D1, NOTHING, 
     $                  Z, LDZ, WORK, RESU(1) )
         ELSEIF(MATYPE.EQ.3)THEN
            UPLO='L'
            CALL DSYT21( 1, UPLO, N, 0, A, LDA, D1, NOTHING, Z, LDZ,
     $               NOTHING, LDZ, NOTHING, WORK, RESU( 1 ) )
         ENDIF
C         WRITE(unit=6,FMT='(A)'),"DONEEEEEEEEEEEEEEEEE"
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


         GOTO 300
*        ===============================
*        NOT IN USE 
*        ===============================
*        compuyte norm of A
         ANORM = MAX( DLANSY( '1', 'L', N, A, LDA, WORK ), UNFL )
*        COPY Z into WORK
         CALL DLACPY( 'A', N, N, Z, LDZ, WORK, N )
*        Compute LAMDA_i * Z_i 
         DO I =1,N
            CALL DSCAL(N, D1(I), WORK( (I-1)*N +1 ),1)
         ENDDO
*        COMPUTE A*zi - Lamda_i zi
         CALL DGEMM('N','N',N, N, N, ONE, A, LDA, Z, LDZ, MONE, WORK, N)
*        Now WORK is a matrix whose column are A*zi - Lamda_i zi
         MAXABS_1 = ZERO
         MAXABS_2 = ZERO
         MAXNORM  = ZERO
         MAXDOT   = ZERO
         DO I =1,N
            J = IDAMAX(N, WORK( (I-1)*N +1 ) , 1)
            MAXABS_1 = MAX( WORK( (I-1)*N + J), MAXABS_1)
            MAXNORM  = MAX( DNRM2(N, WORK( (I-1)*N +1 ), 1), MAXNORM)
         ENDDO
*        LOSS ORTH TEST
         DO I =1,N
            DO J=I+1,N
               MAXDOT = MAX( DDOT(N, Z(1,I), 1, Z(1,J), 1), MAXDOT)
            ENDDO
         ENDDO
     
         RESU(4) = MAXABS_1
         RESU(5) = MAXNORM
         RESU(6) = MAXDOT
     
C         WRITE(unit=6,FMT='(A)')
C     $' ================================================================
C     $============================================='
C         WRITE(unit=6,FMT='(A,E15.5)') 
C     $   " | Azi - Di zi |                    ", MAXABS_1
C         WRITE(unit=6,FMT='(A,E15.5)') 
C     $   " ||Azi - Di zi||                    ", MAXNORM
C         WRITE(unit=6,FMT='(A,E15.5)') 
C     $   " | zi*zj' |                         ", MAXDOT

      ENDIF



      






 300  CONTINUE
      RETURN
      END

