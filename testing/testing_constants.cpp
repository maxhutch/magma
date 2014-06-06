/*
    -- MAGMA (version 1.5.0-beta1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date April 2014

       @author Mark Gates
*/
#include <stdio.h>

#undef NDEBUG
#include <assert.h>

#include <cublas_v2.h>

#include "magma.h"

int main( int argc, char** argv )
{
    // ------------------------------------------------------------
    printf( "testing MAGMA  -> lapack_xxxxx_const\n" );
    assert( lapack_bool_const(   MagmaFalse         )[0] == 'N' );
    assert( lapack_bool_const(   MagmaTrue          )[0] == 'Y' );

    assert( lapack_order_const(  MagmaRowMajor      )[0] == 'R' );
    assert( lapack_order_const(  MagmaColMajor      )[0] == 'C' );

    assert( lapack_trans_const(  MagmaNoTrans       )[0] == 'N' );
    assert( lapack_trans_const(  MagmaTrans         )[0] == 'T' );
    assert( lapack_trans_const(  MagmaConjTrans     )[0] == 'C' );

    assert( lapack_uplo_const(   MagmaUpper         )[0] == 'U' );
    assert( lapack_uplo_const(   MagmaLower         )[0] == 'L' );
    assert( lapack_uplo_const(   MagmaFull          )[0] == 'F' );

    assert( lapack_diag_const(   MagmaNonUnit       )[0] == 'N' );
    assert( lapack_diag_const(   MagmaUnit          )[0] == 'U' );

    assert( lapack_side_const(   MagmaLeft          )[0] == 'L' );
    assert( lapack_side_const(   MagmaRight         )[0] == 'R' );
    assert( lapack_side_const(   MagmaBothSides     )[0] == 'B' );

    assert( lapack_norm_const(   MagmaOneNorm       )[0] == '1' );
    assert( lapack_norm_const(   MagmaTwoNorm       )[0] == '2' );
    assert( lapack_norm_const(   MagmaFrobeniusNorm )[0] == 'F' );
    assert( lapack_norm_const(   MagmaInfNorm       )[0] == 'I' );
    assert( lapack_norm_const(   MagmaMaxNorm       )[0] == 'M' );

    assert( lapack_dist_const(   MagmaDistUniform   )[0] == 'U' );
    assert( lapack_dist_const(   MagmaDistSymmetric )[0] == 'S' );
    assert( lapack_dist_const(   MagmaDistNormal    )[0] == 'N' );

    assert( lapack_sym_const(    MagmaHermGeev      )[0] == 'H' );
    assert( lapack_sym_const(    MagmaHermPoev      )[0] == 'P' );
    assert( lapack_sym_const(    MagmaNonsymPosv    )[0] == 'N' );
    assert( lapack_sym_const(    MagmaSymPosv       )[0] == 'S' );

    assert( lapack_pack_const(   MagmaNoPacking     )[0] == 'N' );
    assert( lapack_pack_const(   MagmaPackSubdiag   )[0] == 'U' );
    assert( lapack_pack_const(   MagmaPackSupdiag   )[0] == 'L' );
    assert( lapack_pack_const(   MagmaPackColumn    )[0] == 'C' );
    assert( lapack_pack_const(   MagmaPackRow       )[0] == 'R' );
    assert( lapack_pack_const(   MagmaPackLowerBand )[0] == 'B' );
    assert( lapack_pack_const(   MagmaPackUpeprBand )[0] == 'Q' );
    assert( lapack_pack_const(   MagmaPackAll       )[0] == 'Z' );

    assert( lapack_vec_const(    MagmaNoVec         )[0] == 'N' );
    assert( lapack_vec_const(    MagmaVec           )[0] == 'V' );
    assert( lapack_vec_const(    MagmaIVec          )[0] == 'I' );
    assert( lapack_vec_const(    MagmaAllVec        )[0] == 'A' );
    assert( lapack_vec_const(    MagmaSomeVec       )[0] == 'S' );
    assert( lapack_vec_const(    MagmaOverwriteVec  )[0] == 'O' );

    assert( lapack_range_const(  MagmaRangeAll      )[0] == 'A' );
    assert( lapack_range_const(  MagmaRangeV        )[0] == 'V' );
    assert( lapack_range_const(  MagmaRangeI        )[0] == 'I' );

    assert( lapack_direct_const( MagmaForward       )[0] == 'F' );
    assert( lapack_direct_const( MagmaBackward      )[0] == 'B' );

    assert( lapack_storev_const( MagmaColumnwise    )[0] == 'C' );
    assert( lapack_storev_const( MagmaRowwise       )[0] == 'R' );


    // ------------------------------------------------------------
    printf( "testing MAGMA  -> lapacke_xxxxx_const\n" );
    assert( lapacke_bool_const(   MagmaFalse         ) == 'N' );
    assert( lapacke_bool_const(   MagmaTrue          ) == 'Y' );

    assert( lapacke_order_const(  MagmaRowMajor      ) == 'R' );
    assert( lapacke_order_const(  MagmaColMajor      ) == 'C' );

    assert( lapacke_trans_const(  MagmaNoTrans       ) == 'N' );
    assert( lapacke_trans_const(  MagmaTrans         ) == 'T' );
    assert( lapacke_trans_const(  MagmaConjTrans     ) == 'C' );

    assert( lapacke_uplo_const(   MagmaUpper         ) == 'U' );
    assert( lapacke_uplo_const(   MagmaLower         ) == 'L' );
    assert( lapacke_uplo_const(   MagmaFull          ) == 'F' );

    assert( lapacke_diag_const(   MagmaNonUnit       ) == 'N' );
    assert( lapacke_diag_const(   MagmaUnit          ) == 'U' );

    assert( lapacke_side_const(   MagmaLeft          ) == 'L' );
    assert( lapacke_side_const(   MagmaRight         ) == 'R' );
    assert( lapacke_side_const(   MagmaBothSides     ) == 'B' );

    assert( lapacke_norm_const(   MagmaOneNorm       ) == '1' );
    assert( lapacke_norm_const(   MagmaTwoNorm       ) == '2' );
    assert( lapacke_norm_const(   MagmaFrobeniusNorm ) == 'F' );
    assert( lapacke_norm_const(   MagmaInfNorm       ) == 'I' );
    assert( lapacke_norm_const(   MagmaMaxNorm       ) == 'M' );

    assert( lapacke_dist_const(   MagmaDistUniform   ) == 'U' );
    assert( lapacke_dist_const(   MagmaDistSymmetric ) == 'S' );
    assert( lapacke_dist_const(   MagmaDistNormal    ) == 'N' );

    assert( lapacke_sym_const(    MagmaHermGeev      ) == 'H' );
    assert( lapacke_sym_const(    MagmaHermPoev      ) == 'P' );
    assert( lapacke_sym_const(    MagmaNonsymPosv    ) == 'N' );
    assert( lapacke_sym_const(    MagmaSymPosv       ) == 'S' );

    assert( lapacke_pack_const(   MagmaNoPacking     ) == 'N' );
    assert( lapacke_pack_const(   MagmaPackSubdiag   ) == 'U' );
    assert( lapacke_pack_const(   MagmaPackSupdiag   ) == 'L' );
    assert( lapacke_pack_const(   MagmaPackColumn    ) == 'C' );
    assert( lapacke_pack_const(   MagmaPackRow       ) == 'R' );
    assert( lapacke_pack_const(   MagmaPackLowerBand ) == 'B' );
    assert( lapacke_pack_const(   MagmaPackUpeprBand ) == 'Q' );
    assert( lapacke_pack_const(   MagmaPackAll       ) == 'Z' );

    assert( lapacke_vec_const(    MagmaNoVec         ) == 'N' );
    assert( lapacke_vec_const(    MagmaVec           ) == 'V' );
    assert( lapacke_vec_const(    MagmaIVec          ) == 'I' );
    assert( lapacke_vec_const(    MagmaAllVec        ) == 'A' );
    assert( lapacke_vec_const(    MagmaSomeVec       ) == 'S' );
    assert( lapacke_vec_const(    MagmaOverwriteVec  ) == 'O' );

    assert( lapacke_range_const(  MagmaRangeAll      ) == 'A' );
    assert( lapacke_range_const(  MagmaRangeV        ) == 'V' );
    assert( lapacke_range_const(  MagmaRangeI        ) == 'I' );

    assert( lapacke_direct_const( MagmaForward       ) == 'F' );
    assert( lapacke_direct_const( MagmaBackward      ) == 'B' );

    assert( lapacke_storev_const( MagmaColumnwise    ) == 'C' );
    assert( lapacke_storev_const( MagmaRowwise       ) == 'R' );


    // ------------------------------------------------------------
    printf( "testing MAGMA  -> lapack_const\n" );
    assert( lapack_const( MagmaFalse         )[0] == 'N' );
    assert( lapack_const( MagmaTrue          )[0] == 'Y' );

    assert( lapack_const( MagmaRowMajor      )[0] == 'R' );
    assert( lapack_const( MagmaColMajor      )[0] == 'C' );

    assert( lapack_const( MagmaNoTrans       )[0] == 'N' );
    assert( lapack_const( MagmaTrans         )[0] == 'T' );
    assert( lapack_const( MagmaConjTrans     )[0] == 'C' );

    assert( lapack_const( MagmaUpper         )[0] == 'U' );
    assert( lapack_const( MagmaLower         )[0] == 'L' );
    assert( lapack_const( MagmaFull          )[0] == 'F' );

    assert( lapack_const( MagmaNonUnit       )[0] == 'N' );
    assert( lapack_const( MagmaUnit          )[0] == 'U' );

    assert( lapack_const( MagmaLeft          )[0] == 'L' );
    assert( lapack_const( MagmaRight         )[0] == 'R' );
    assert( lapack_const( MagmaBothSides     )[0] == 'B' );

    assert( lapack_const( MagmaOneNorm       )[0] == '1' );
    assert( lapack_const( MagmaTwoNorm       )[0] == '2' );
    assert( lapack_const( MagmaFrobeniusNorm )[0] == 'F' );
    assert( lapack_const( MagmaInfNorm       )[0] == 'I' );
    assert( lapack_const( MagmaMaxNorm       )[0] == 'M' );

    assert( lapack_const( MagmaDistUniform   )[0] == 'U' );
    assert( lapack_const( MagmaDistSymmetric )[0] == 'S' );
    assert( lapack_const( MagmaDistNormal    )[0] == 'N' );

    assert( lapack_const( MagmaHermGeev      )[0] == 'H' );
    assert( lapack_const( MagmaHermPoev      )[0] == 'P' );
    assert( lapack_const( MagmaNonsymPosv    )[0] == 'N' );
    assert( lapack_const( MagmaSymPosv       )[0] == 'S' );

    assert( lapack_const( MagmaNoPacking     )[0] == 'N' );
    assert( lapack_const( MagmaPackSubdiag   )[0] == 'U' );
    assert( lapack_const( MagmaPackSupdiag   )[0] == 'L' );
    assert( lapack_const( MagmaPackColumn    )[0] == 'C' );
    assert( lapack_const( MagmaPackRow       )[0] == 'R' );
    assert( lapack_const( MagmaPackLowerBand )[0] == 'B' );
    assert( lapack_const( MagmaPackUpeprBand )[0] == 'Q' );
    assert( lapack_const( MagmaPackAll       )[0] == 'Z' );

    assert( lapack_const( MagmaNoVec         )[0] == 'N' );
    assert( lapack_const( MagmaVec           )[0] == 'V' );
    assert( lapack_const( MagmaIVec          )[0] == 'I' );
    assert( lapack_const( MagmaAllVec        )[0] == 'A' );
    assert( lapack_const( MagmaSomeVec       )[0] == 'S' );
    assert( lapack_const( MagmaOverwriteVec  )[0] == 'O' );

    assert( lapack_const( MagmaRangeAll      )[0] == 'A' );
    assert( lapack_const( MagmaRangeV        )[0] == 'V' );
    assert( lapack_const( MagmaRangeI        )[0] == 'I' );

    assert( lapack_const( MagmaForward       )[0] == 'F' );
    assert( lapack_const( MagmaBackward      )[0] == 'B' );

    assert( lapack_const( MagmaColumnwise    )[0] == 'C' );
    assert( lapack_const( MagmaRowwise       )[0] == 'R' );


    // ------------------------------------------------------------
    printf( "testing MAGMA  -> lapacke_const\n" );
    assert( lapacke_const( MagmaFalse         ) == 'N' );
    assert( lapacke_const( MagmaTrue          ) == 'Y' );

    assert( lapacke_const( MagmaRowMajor      ) == 'R' );
    assert( lapacke_const( MagmaColMajor      ) == 'C' );

    assert( lapacke_const( MagmaNoTrans       ) == 'N' );
    assert( lapacke_const( MagmaTrans         ) == 'T' );
    assert( lapacke_const( MagmaConjTrans     ) == 'C' );

    assert( lapacke_const( MagmaUpper         ) == 'U' );
    assert( lapacke_const( MagmaLower         ) == 'L' );
    assert( lapacke_const( MagmaFull          ) == 'F' );

    assert( lapacke_const( MagmaNonUnit       ) == 'N' );
    assert( lapacke_const( MagmaUnit          ) == 'U' );

    assert( lapacke_const( MagmaLeft          ) == 'L' );
    assert( lapacke_const( MagmaRight         ) == 'R' );
    assert( lapacke_const( MagmaBothSides     ) == 'B' );

    assert( lapacke_const( MagmaOneNorm       ) == '1' );
    assert( lapacke_const( MagmaTwoNorm       ) == '2' );
    assert( lapacke_const( MagmaFrobeniusNorm ) == 'F' );
    assert( lapacke_const( MagmaInfNorm       ) == 'I' );
    assert( lapacke_const( MagmaMaxNorm       ) == 'M' );

    assert( lapacke_const( MagmaDistUniform   ) == 'U' );
    assert( lapacke_const( MagmaDistSymmetric ) == 'S' );
    assert( lapacke_const( MagmaDistNormal    ) == 'N' );

    assert( lapacke_const( MagmaHermGeev      ) == 'H' );
    assert( lapacke_const( MagmaHermPoev      ) == 'P' );
    assert( lapacke_const( MagmaNonsymPosv    ) == 'N' );
    assert( lapacke_const( MagmaSymPosv       ) == 'S' );

    assert( lapacke_const( MagmaNoPacking     ) == 'N' );
    assert( lapacke_const( MagmaPackSubdiag   ) == 'U' );
    assert( lapacke_const( MagmaPackSupdiag   ) == 'L' );
    assert( lapacke_const( MagmaPackColumn    ) == 'C' );
    assert( lapacke_const( MagmaPackRow       ) == 'R' );
    assert( lapacke_const( MagmaPackLowerBand ) == 'B' );
    assert( lapacke_const( MagmaPackUpeprBand ) == 'Q' );
    assert( lapacke_const( MagmaPackAll       ) == 'Z' );

    assert( lapacke_const( MagmaNoVec         ) == 'N' );
    assert( lapacke_const( MagmaVec           ) == 'V' );
    assert( lapacke_const( MagmaIVec          ) == 'I' );
    assert( lapacke_const( MagmaAllVec        ) == 'A' );
    assert( lapacke_const( MagmaSomeVec       ) == 'S' );
    assert( lapacke_const( MagmaOverwriteVec  ) == 'O' );

    assert( lapacke_const( MagmaRangeAll      ) == 'A' );
    assert( lapacke_const( MagmaRangeV        ) == 'V' );
    assert( lapacke_const( MagmaRangeI        ) == 'I' );

    assert( lapacke_const( MagmaForward       ) == 'F' );
    assert( lapacke_const( MagmaBackward      ) == 'B' );

    assert( lapacke_const( MagmaColumnwise    ) == 'C' );
    assert( lapacke_const( MagmaRowwise       ) == 'R' );


    // ------------------------------------------------------------
    printf( "testing LAPACK -> magma_xxxxx_const\n" );
    assert( magma_bool_const('N') == MagmaFalse );
    assert( magma_bool_const('n') == MagmaFalse );
    assert( magma_bool_const('Y') == MagmaTrue  );
    assert( magma_bool_const('y') == MagmaTrue  );

    assert( magma_order_const( 'R' ) == MagmaRowMajor  );
    assert( magma_order_const( 'r' ) == MagmaRowMajor  );
    assert( magma_order_const( 'C' ) == MagmaColMajor  );
    assert( magma_order_const( 'c' ) == MagmaColMajor  );

    assert( magma_trans_const( 'N' ) == MagmaNoTrans   );
    assert( magma_trans_const( 'n' ) == MagmaNoTrans   );
    assert( magma_trans_const( 'T' ) == MagmaTrans     );
    assert( magma_trans_const( 't' ) == MagmaTrans     );
    assert( magma_trans_const( 'C' ) == MagmaConjTrans );
    assert( magma_trans_const( 'c' ) == MagmaConjTrans );

    assert( magma_uplo_const( 'U' ) == MagmaUpper      );
    assert( magma_uplo_const( 'u' ) == MagmaUpper      );
    assert( magma_uplo_const( 'L' ) == MagmaLower      );
    assert( magma_uplo_const( 'l' ) == MagmaLower      );
    assert( magma_uplo_const( 'A' ) == MagmaFull       );  // anything else
    assert( magma_uplo_const( 'a' ) == MagmaFull       );
    assert( magma_uplo_const( 'G' ) == MagmaFull       );
    assert( magma_uplo_const( 'g' ) == MagmaFull       );
    assert( magma_uplo_const( 'F' ) == MagmaFull       );
    assert( magma_uplo_const( 'f' ) == MagmaFull       );

    assert( magma_diag_const( 'N' ) == MagmaNonUnit    );
    assert( magma_diag_const( 'n' ) == MagmaNonUnit    );
    assert( magma_diag_const( 'U' ) == MagmaUnit       );
    assert( magma_diag_const( 'u' ) == MagmaUnit       );

    assert( magma_side_const( 'L' ) == MagmaLeft       );
    assert( magma_side_const( 'l' ) == MagmaLeft       );
    assert( magma_side_const( 'R' ) == MagmaRight      );
    assert( magma_side_const( 'r' ) == MagmaRight      );

    assert( magma_norm_const( 'O' ) == MagmaOneNorm       );
    assert( magma_norm_const( 'o' ) == MagmaOneNorm       );
    assert( magma_norm_const( '1' ) == MagmaOneNorm       );
    assert( magma_norm_const( '2' ) == MagmaTwoNorm       );
    assert( magma_norm_const( 'F' ) == MagmaFrobeniusNorm );
    assert( magma_norm_const( 'f' ) == MagmaFrobeniusNorm );
    assert( magma_norm_const( 'E' ) == MagmaFrobeniusNorm );
    assert( magma_norm_const( 'e' ) == MagmaFrobeniusNorm );
    assert( magma_norm_const( 'I' ) == MagmaInfNorm       );
    assert( magma_norm_const( 'i' ) == MagmaInfNorm       );
    assert( magma_norm_const( 'M' ) == MagmaMaxNorm       );
    assert( magma_norm_const( 'm' ) == MagmaMaxNorm       );

    assert( magma_dist_const( 'U' ) == MagmaDistUniform   );
    assert( magma_dist_const( 'u' ) == MagmaDistUniform   );
    assert( magma_dist_const( 'S' ) == MagmaDistSymmetric );
    assert( magma_dist_const( 's' ) == MagmaDistSymmetric );
    assert( magma_dist_const( 'N' ) == MagmaDistNormal    );
    assert( magma_dist_const( 'n' ) == MagmaDistNormal    );

    //assert( magma_xxxx_const( 'H' ) == MagmaHermGeev      );
    //assert( magma_xxxx_const( 'P' ) == MagmaHermPoev      );
    //assert( magma_xxxx_const( 'N' ) == MagmaNonsymPosv    );
    //assert( magma_xxxx_const( 'S' ) == MagmaSymPosv       );

    assert( magma_pack_const( 'N' ) == MagmaNoPacking     );
    assert( magma_pack_const( 'n' ) == MagmaNoPacking     );
    assert( magma_pack_const( 'U' ) == MagmaPackSubdiag   );
    assert( magma_pack_const( 'u' ) == MagmaPackSubdiag   );
    assert( magma_pack_const( 'L' ) == MagmaPackSupdiag   );
    assert( magma_pack_const( 'l' ) == MagmaPackSupdiag   );
    assert( magma_pack_const( 'C' ) == MagmaPackColumn    );
    assert( magma_pack_const( 'c' ) == MagmaPackColumn    );
    assert( magma_pack_const( 'R' ) == MagmaPackRow       );
    assert( magma_pack_const( 'r' ) == MagmaPackRow       );
    assert( magma_pack_const( 'B' ) == MagmaPackLowerBand );
    assert( magma_pack_const( 'b' ) == MagmaPackLowerBand );
    assert( magma_pack_const( 'Q' ) == MagmaPackUpeprBand );
    assert( magma_pack_const( 'q' ) == MagmaPackUpeprBand );
    assert( magma_pack_const( 'Z' ) == MagmaPackAll       );
    assert( magma_pack_const( 'z' ) == MagmaPackAll       );

    assert( magma_vec_const( 'N' )  == MagmaNoVec         );
    assert( magma_vec_const( 'n' )  == MagmaNoVec         );
    assert( magma_vec_const( 'V' )  == MagmaVec           );
    assert( magma_vec_const( 'v' )  == MagmaVec           );
    assert( magma_vec_const( 'I' )  == MagmaIVec          );
    assert( magma_vec_const( 'i' )  == MagmaIVec          );
    assert( magma_vec_const( 'A' )  == MagmaAllVec        );
    assert( magma_vec_const( 'a' )  == MagmaAllVec        );
    assert( magma_vec_const( 'S' )  == MagmaSomeVec       );
    assert( magma_vec_const( 's' )  == MagmaSomeVec       );
    assert( magma_vec_const( 'O' )  == MagmaOverwriteVec  );
    assert( magma_vec_const( 'o' )  == MagmaOverwriteVec  );

    assert( magma_range_const( 'A' )  == MagmaRangeAll    );
    assert( magma_range_const( 'a' )  == MagmaRangeAll    );
    assert( magma_range_const( 'V' )  == MagmaRangeV      );
    assert( magma_range_const( 'v' )  == MagmaRangeV      );
    assert( magma_range_const( 'I' )  == MagmaRangeI      );
    assert( magma_range_const( 'i' )  == MagmaRangeI      );

    assert( magma_direct_const( 'F' ) == MagmaForward     );
    assert( magma_direct_const( 'f' ) == MagmaForward     );
    assert( magma_direct_const( 'B' ) == MagmaBackward    );
    assert( magma_direct_const( 'b' ) == MagmaBackward    );

    assert( magma_storev_const( 'C' ) == MagmaColumnwise  );
    assert( magma_storev_const( 'c' ) == MagmaColumnwise  );
    assert( magma_storev_const( 'R' ) == MagmaRowwise     );
    assert( magma_storev_const( 'r' ) == MagmaRowwise     );


    // ------------------------------------------------------------
    #ifdef HAVE_clAmdBlas
    printf( "testing MAGMA  -> amdblas_xxxxx_const\n" );
    assert( amdblas_order_const( MagmaRowMajor      ) == clAmdBlasRowMajor    );
    assert( amdblas_order_const( MagmaColMajor      ) == clAmdBlasColumnMajor );

    assert( amdblas_trans_const( MagmaNoTrans       ) == clAmdBlasNoTrans     );
    assert( amdblas_trans_const( MagmaTrans         ) == clAmdBlasTrans       );
    assert( amdblas_trans_const( MagmaConjTrans     ) == clAmdBlasConjTrans   );

    assert( amdblas_uplo_const(  MagmaUpper         ) == clAmdBlasUpper       );
    assert( amdblas_uplo_const(  MagmaLower         ) == clAmdBlasLower       );

    assert( amdblas_diag_const(  MagmaNonUnit       ) == clAmdBlasNonUnit     );
    assert( amdblas_diag_const(  MagmaUnit          ) == clAmdBlasUnit        );

    assert( amdblas_side_const(  MagmaLeft          ) == clAmdBlasLeft        );
    assert( amdblas_side_const(  MagmaRight         ) == clAmdBlasRight       );
    #endif


    // ------------------------------------------------------------
    #ifdef CUBLAS_V2_H_
    printf( "testing MAGMA  -> cublas_xxxxx_const\n" );
    assert( cublas_trans_const( MagmaNoTrans       ) == CUBLAS_OP_N            );
    assert( cublas_trans_const( MagmaTrans         ) == CUBLAS_OP_T            );
    assert( cublas_trans_const( MagmaConjTrans     ) == CUBLAS_OP_C            );

    assert( cublas_uplo_const(  MagmaUpper         ) == CUBLAS_FILL_MODE_UPPER );
    assert( cublas_uplo_const(  MagmaLower         ) == CUBLAS_FILL_MODE_LOWER );

    assert( cublas_diag_const(  MagmaNonUnit       ) == CUBLAS_DIAG_NON_UNIT   );
    assert( cublas_diag_const(  MagmaUnit          ) == CUBLAS_DIAG_UNIT       );

    assert( cublas_side_const(  MagmaLeft          ) == CUBLAS_SIDE_LEFT       );
    assert( cublas_side_const(  MagmaRight         ) == CUBLAS_SIDE_RIGHT      );
    #endif


    // ------------------------------------------------------------
    #ifdef HAVE_CBLAS
    printf( "testing MAGMA  -> cblas_xxxxx_const\n" );
    assert( cblas_order_const( MagmaRowMajor      ) == CblasRowMajor  );
    assert( cblas_order_const( MagmaColMajor      ) == CblasColMajor  );

    assert( cblas_trans_const( MagmaNoTrans       ) == CblasNoTrans   );
    assert( cblas_trans_const( MagmaTrans         ) == CblasTrans     );
    assert( cblas_trans_const( MagmaConjTrans     ) == CblasConjTrans );

    assert( cblas_uplo_const(  MagmaUpper         ) == CblasUpper     );
    assert( cblas_uplo_const(  MagmaLower         ) == CblasLower     );

    assert( cblas_diag_const(  MagmaNonUnit       ) == CblasNonUnit   );
    assert( cblas_diag_const(  MagmaUnit          ) == CblasUnit      );

    assert( cblas_side_const(  MagmaLeft          ) == CblasLeft      );
    assert( cblas_side_const(  MagmaRight         ) == CblasRight     );
    #endif

    //assert( true  );
    //assert( false );

    return 0;
}
