!
!   -- MAGMA (version 1.6.1) --
!      Univ. of Tennessee, Knoxville
!      Univ. of California, Berkeley
!      Univ. of Colorado, Denver
!      @date January 2015
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
  
  ! parameter constants from magma_types.h
  integer, parameter :: &
        MagmaFalse         = 0,    &
        MagmaTrue          = 1,    &
        MagmaRowMajor      = 101,  &
        MagmaColMajor      = 102,  &
        MagmaNoTrans       = 111,  &
        MagmaTrans         = 112,  &
        MagmaConjTrans     = 113,  &
        MagmaUpper         = 121,  &
        MagmaLower         = 122,  &
        MagmaUpperLower    = 123,  &
        MagmaFull          = 123,  &
        MagmaNonUnit       = 131,  &
        MagmaUnit          = 132,  &
        MagmaLeft          = 141,  &
        MagmaRight         = 142,  &
        MagmaBothSides     = 143,  &
        MagmaOneNorm       = 171,  &
        MagmaRealOneNorm   = 172,  &
        MagmaTwoNorm       = 173,  &
        MagmaFrobeniusNorm = 174,  &
        MagmaInfNorm       = 175,  &
        MagmaRealInfNorm   = 176,  &
        MagmaMaxNorm       = 177,  &
        MagmaRealMaxNorm   = 178,  &
        MagmaDistUniform   = 201,  &
        MagmaDistSymmetric = 202,  &
        MagmaDistNormal    = 203,  &
        MagmaHermGeev      = 241,  &
        MagmaHermPoev      = 242,  &
        MagmaNonsymPosv    = 243,  &
        MagmaSymPosv       = 244,  &
        MagmaNoPacking     = 291,  &
        MagmaPackSubdiag   = 292,  &
        MagmaPackSupdiag   = 293,  &
        MagmaPackColumn    = 294,  &
        MagmaPackRow       = 295,  &
        MagmaPackLowerBand = 296,  &
        MagmaPackUpeprBand = 297,  &
        MagmaPackAll       = 298,  &
        MagmaNoVec         = 301,  &
        MagmaVec           = 302,  &
        MagmaIVec          = 303,  &
        MagmaAllVec        = 304,  &
        MagmaSomeVec       = 305,  &
        MagmaOverwriteVec  = 306,  &
        MagmaBacktransVec  = 307,  &
        MagmaRangeAll      = 311,  &
        MagmaRangeV        = 312,  &
        MagmaRangeI        = 313,  &
        MagmaQ             = 322,  &
        MagmaP             = 323,  &
        MagmaForward       = 391,  &
        MagmaBackward      = 392,  &
        MagmaColumnwise    = 401,  &
        MagmaRowwise       = 402

end module magma
