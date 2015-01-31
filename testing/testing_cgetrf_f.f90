!
!   -- MAGMA (version 1.6.1) --
!      Univ. of Tennessee, Knoxville
!      Univ. of California, Berkeley
!      Univ. of Colorado, Denver
!      @date January 2015
!
!  @generated from testing_zgetrf_f.f90 normal z -> c, Fri Jan 30 19:00:26 2015
!
      program testing_cgetrf_f

      use magma

      external slamch, clange, cgemm, cgesv, cgetrs

      real clange, slamch

      real              :: rnumber(2), Anorm, Bnorm, Xnorm, Rnorm 
      real, allocatable :: work(:)
      complex, allocatable       :: A(:), B(:), X(:)
      complex, allocatable       :: A2(:)
      integer,    allocatable       :: ipiv(:)

      complex                    :: zone, mzone
      integer                       :: i, n, info, lda
      integer                       :: nrhs
      real(kind=8)                  :: flops, t, tstart, tend

      PARAMETER          ( nrhs = 1, zone = 1., mzone = -1. )
      
      call cublas_init()

      n   = 2048
      lda = n
 
!------ Allocate CPU memory
      allocate(A(lda*n))
      allocate(A2(lda*n))
      allocate(B(lda*nrhs))
      allocate(X(lda*nrhs))
      allocate(ipiv(n))
      allocate(work(n))

!---- Initializa the matrix
      do i=1,n*n
         call random_number(rnumber)
         A(i) = rnumber(1)
      end do
      A2(:) = A(:)

      do i=1,n*nrhs
         call random_number(rnumber)
         B(i) = rnumber(1)
      end do
      X(:) = B(:)

!---- Call magma LU ----------------
      call magma_wtime_f(tstart)
      call magmaf_cgetrf(n, n, A, lda, ipiv, info)
      call magma_wtime_f(tend)

      if ( info .ne. 0 )  then
         write(*,*) "Info : ", info
      end if

!---- Call solve -------------
      call cgetrs('n', n, nrhs, A, lda, ipiv, X, lda, info)

      if ( info .ne. 0 )  then
         write(*,*) "Info : ", info
      end if

!---- Compare the two results ------
      Anorm = clange('I', n, n,    A2, lda, work)
      Bnorm = clange('I', n, nrhs, B,  lda, work)
      Xnorm = clange('I', n, nrhs, X,  lda, work)

      call cgemm('n', 'n', n,  nrhs, n, zone, A2, lda, X, lda, mzone, B, lda)
      Rnorm = clange('I', n, nrhs, B, lda, work)
            
      write(*,*)
      write(*,*  ) 'Solving A x = b using LU factorization:'
      write(*,105) '  || A || = ', Anorm
      write(*,105) '  || b || = ', Bnorm
      write(*,105) '  || x || = ', Xnorm
      write(*,105) '  || b - A x || = ', Rnorm
      
      flops = 2. * n * n * n / 3.
      t = tend - tstart

      write(*,*)   '  Gflops  = ',  flops / t / 1e9
      write(*,*)

      Rnorm = Rnorm / ( (Anorm*Xnorm+Bnorm) * n * slamch('E') )

      if ( Rnorm > 60. ) then
         write(*,105) '  Solution is suspicious, ', Rnorm
      else
         write(*,105) '  Solution is CORRECT' 
      end if

!---- Free CPU memory
      deallocate(A, A2, B, X, ipiv, work)

!---- Free GPU memory
      call cublas_shutdown()

 105  format((a35,es10.3))

      end
