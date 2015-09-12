!
!   -- MAGMA (version 1.7.0) --
!      Univ. of Tennessee, Knoxville
!      Univ. of California, Berkeley
!      Univ. of Colorado, Denver
!      @date September 2015
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

end module magma
