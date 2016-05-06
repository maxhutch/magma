!
!   -- MAGMA (version 2.0.2) --
!      Univ. of Tennessee, Knoxville
!      Univ. of California, Berkeley
!      Univ. of Colorado, Denver
!      @date May 2016
!

module magma

    use magma_param
    use magma_zfortran
    use magma_dfortran
    use magma_cfortran
    use magma_sfortran
    
    !---- Fortran interfaces to MAGMA subroutines ----
    interface
    
    subroutine magmaf_init( )
    end subroutine
    
    subroutine magmaf_finalize(  )
    end subroutine
    
    subroutine magmaf_wtime( time )
        double precision :: time
    end subroutine
    
    end interface
    
    ! parameter constants from magma_types.h
    ! currently MAGMA's Fortran interface uses characters, not integers
    character, parameter :: &
        MagmaFalse         = 'n',  &
        MagmaTrue          = 'y',  &
        MagmaRowMajor      = 'r',  &
        MagmaColMajor      = 'c',  &
        MagmaNoTrans       = 'n',  &
        MagmaTrans         = 't',  &
        MagmaConjTrans     = 'c',  &
        MagmaUpper         = 'u',  &
        MagmaLower         = 'l',  &
        MagmaFull          = 'f',  &
        MagmaNonUnit       = 'n',  &
        MagmaUnit          = 'u',  &
        MagmaLeft          = 'l',  &
        MagmaRight         = 'r',  &
        MagmaBothSides     = 'b',  &
        MagmaOneNorm       = '1',  &
        MagmaTwoNorm       = '2',  &
        MagmaFrobeniusNorm = 'f',  &
        MagmaInfNorm       = 'i',  &
        MagmaMaxNorm       = 'm',  &
        MagmaDistUniform   = 'u',  &
        MagmaDistSymmetric = 's',  &
        MagmaDistNormal    = 'n',  &
        MagmaHermGeev      = 'h',  &
        MagmaHermPoev      = 'p',  &
        MagmaNonsymPosv    = 'n',  &
        MagmaSymPosv       = 's',  &
        MagmaNoPacking     = 'n',  &
        MagmaPackSubdiag   = 'u',  &
        MagmaPackSupdiag   = 'l',  &
        MagmaPackColumn    = 'c',  &
        MagmaPackRow       = 'r',  &
        MagmaPackLowerBand = 'b',  &
        MagmaPackUpeprBand = 'q',  &
        MagmaPackAll       = 'z',  &
        MagmaNoVec         = 'n',  &
        MagmaVec           = 'v',  &
        MagmaIVec          = 'i',  &
        MagmaAllVec        = 'a',  &
        MagmaSomeVec       = 's',  &
        MagmaOverwriteVec  = 'o',  &
        MagmaBacktransVec  = 'b',  &
        MagmaRangeAll      = 'a',  &
        MagmaRangeV        = 'v',  &
        MagmaRangeI        = 'i',  &
        MagmaQ             = 'q',  &
        MagmaP             = 'p',  &
        MagmaForward       = 'f',  &
        MagmaBackward      = 'b',  &
        MagmaColumnwise    = 'c',  &
        MagmaRowwise       = 'r'

contains

! --------------------
!> Sets ptrNew = ptrOld( i ), with stride inc.
!! Useful because CUDA pointers are opaque types in Fortran.
subroutine magmaf_soff1d( ptrNew, ptrOld, inc, i )
    magma_devptr_t   :: ptrNew
    magma_devptr_t   :: ptrOld
    integer          :: inc, i

    ptrNew = ptrOld + (i-1) * inc * sizeof_real
end subroutine magmaf_soff1d

!> Sets ptrNew = ptrOld( i, j ), with leading dimension lda.
!! Useful because CUDA pointers are opaque types in Fortran.
subroutine magmaf_soff2d( ptrNew, ptrOld, lda, i, j )
    magma_devptr_t   :: ptrNew
    magma_devptr_t   :: ptrOld
    integer          :: lda, i, j

    ptrNew = ptrOld + ((j-1) * lda + (i-1)) * sizeof_real
end subroutine magmaf_soff2d

! --------------------
!> Sets ptrNew = ptrOld( i ), with stride inc.
!! Useful because CUDA pointers are opaque types in Fortran.
subroutine magmaf_doff1d( ptrNew, ptrOld, inc, i )
    magma_devptr_t   :: ptrNew
    magma_devptr_t   :: ptrOld
    integer          :: inc, i

    ptrNew = ptrOld + (i-1) * inc * sizeof_double
end subroutine magmaf_doff1d

!> Sets ptrNew = ptrOld( i, j ), with leading dimension lda.
!! Useful because CUDA pointers are opaque types in Fortran.
subroutine magmaf_doff2d( ptrNew, ptrOld, lda, i, j )
    magma_devptr_t   :: ptrNew
    magma_devptr_t   :: ptrOld
    integer          :: lda, i, j

    ptrNew = ptrOld + ((j-1) * lda + (i-1)) * sizeof_double
end subroutine magmaf_doff2d

! --------------------
!> Sets ptrNew = ptrOld( i ), with stride inc.
!! Useful because CUDA pointers are opaque types in Fortran.
subroutine magmaf_coff1d( ptrNew, ptrOld, inc, i )
    magma_devptr_t   :: ptrNew
    magma_devptr_t   :: ptrOld
    integer          :: inc, i

    ptrNew = ptrOld + (i-1) * inc * sizeof_complex
end subroutine magmaf_coff1d

!> Sets ptrNew = ptrOld( i, j ), with leading dimension lda.
!! Useful because CUDA pointers are opaque types in Fortran.
subroutine magmaf_coff2d( ptrNew, ptrOld, lda, i, j )
    magma_devptr_t   :: ptrNew
    magma_devptr_t   :: ptrOld
    integer          :: lda, i, j

    ptrNew = ptrOld + ((j-1) * lda + (i-1)) * sizeof_complex
end subroutine magmaf_coff2d

! --------------------
!> Sets ptrNew = ptrOld( i ), with stride inc.
!! Useful because CUDA pointers are opaque types in Fortran.
subroutine magmaf_zoff1d( ptrNew, ptrOld, inc, i )
    magma_devptr_t   :: ptrNew
    magma_devptr_t   :: ptrOld
    integer          :: inc, i

    ptrNew = ptrOld + (i-1) * inc * sizeof_complex_16
end subroutine magmaf_zoff1d

!> Sets ptrNew = ptrOld( i, j ), with leading dimension lda.
!! Useful because CUDA pointers are opaque types in Fortran.
subroutine magmaf_zoff2d( ptrNew, ptrOld, lda, i, j )
    magma_devptr_t   :: ptrNew
    magma_devptr_t   :: ptrOld
    integer          :: lda, i, j

    ptrNew = ptrOld + ((j-1) * lda + (i-1)) * sizeof_complex_16
end subroutine magmaf_zoff2d

! --------------------
!> Sets ptrNew = ptrOld( i ), with stride inc.
!! Useful because CUDA pointers are opaque types in Fortran.
subroutine magmaf_ioff1d( ptrNew, ptrOld, inc, i )
    magma_devptr_t   :: ptrNew
    magma_devptr_t   :: ptrOld
    integer          :: inc, i

    ptrNew = ptrOld + (i-1) * inc * sizeof_integer
end subroutine magmaf_ioff1d

!> Sets ptrNew = ptrOld( i, j ), with leading dimension lda.
!! Useful because CUDA pointers are opaque types in Fortran.
subroutine magmaf_ioff2d( ptrNew, ptrOld, lda, i, j )
    magma_devptr_t   :: ptrNew
    magma_devptr_t   :: ptrOld
    integer          :: lda, i, j

    ptrNew = ptrOld + ((j-1) * lda + (i-1)) * sizeof_integer
end subroutine magmaf_ioff2d

end module magma
