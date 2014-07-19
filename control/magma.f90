!
!   -- MAGMA (version 1.5.0-beta3) --
!      Univ. of Tennessee, Knoxville
!      Univ. of California, Berkeley
!      Univ. of Colorado, Denver
!      @date July 2014
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
