#include <stdio.h>

#include "magma_v2.h"
#include "testings.h"

int main( int argc, char** argv )
{
    magma_init();
    
    magma_opts opts;
    opts.parse_opts( argc, argv );
    
    for( int itest = 0; itest < opts.ntest; ++itest ) {
        printf( "m %5lld, n %5lld, k %5lld\n",
                (long long) opts.msize[itest],
                (long long) opts.nsize[itest],
                (long long) opts.ksize[itest] );
    }
    printf( "\n" );
    
    printf( "ntest    %lld\n", (long long) opts.ntest );
    printf( "\n" );
    
    printf( "nb       %lld\n", (long long) opts.nb       );
    printf( "nrhs     %lld\n", (long long) opts.nrhs     );
    printf( "nqueue   %lld\n", (long long) opts.nqueue   );
    printf( "ngpu     %lld\n", (long long) opts.ngpu     );
    printf( "niter    %lld\n", (long long) opts.niter    );
    printf( "nthread  %lld\n", (long long) opts.nthread  );
    printf( "itype    %lld\n", (long long) opts.itype    );
    printf( "verbose  %lld\n", (long long) opts.verbose  );
    printf( "\n" );
    
    printf( "check    %s\n", (opts.check  ? "true" : "false") );
    printf( "lapack   %s\n", (opts.lapack ? "true" : "false") );
    printf( "warmup   %s\n", (opts.warmup ? "true" : "false") );
    printf( "\n" );
    
    printf( "uplo     %3d (%s)\n", opts.uplo,   lapack_uplo_const(  opts.uplo   ));
    printf( "transA   %3d (%s)\n", opts.transA, lapack_trans_const( opts.transA ));
    printf( "transB   %3d (%s)\n", opts.transB, lapack_trans_const( opts.transB ));
    printf( "side     %3d (%s)\n", opts.side,   lapack_side_const(  opts.side   ));
    printf( "diag     %3d (%s)\n", opts.diag,   lapack_diag_const(  opts.diag   ));
    printf( "jobz     %3d (%s)\n", opts.jobz,   lapack_vec_const(   opts.jobz   ));
    printf( "jobvr    %3d (%s)\n", opts.jobvr,  lapack_vec_const(   opts.jobvr  ));
    printf( "jobvl    %3d (%s)\n", opts.jobvl,  lapack_vec_const(   opts.jobvl  ));
    
    for( auto iter = opts.svd_work.begin(); iter < opts.svd_work.end(); ++iter ) {
        printf( "svd_work %lld\n", (long long) *iter );
    }
    for( auto iter = opts.jobu.begin(); iter < opts.jobu.end(); ++iter ) {
        printf( "jobu     %3d (%s)\n", *iter, lapack_vec_const( *iter ));
    }
    for( auto iter = opts.jobv.begin(); iter < opts.jobv.end(); ++iter ) {
        printf( "jobvt    %3d (%s)\n", *iter, lapack_vec_const( *iter ));
    }
    
    opts.cleanup();
    magma_finalize();
    
    return 0;
}
