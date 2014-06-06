!
!   -- MAGMA (version 1.4.1) --
!      Univ. of Tennessee, Knoxville
!      Univ. of California, Berkeley
!      Univ. of Colorado, Denver
!      December 2013
!

module magma

  use magma_param
  use magma_zfortran
  use magma_dfortran
  use magma_cfortran
  use magma_sfortran

  interface

  subroutine magmaf_init( )
  end subroutine
  
  subroutine magmaf_finalize(  )
  end subroutine
  
  end interface
  
end module magma
