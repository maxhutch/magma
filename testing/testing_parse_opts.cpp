#include "testings.h"

int main( int argc, char** argv )
{
    magma_opts opts;
    parse_opts( argc, argv, &opts );
    
    for( int i = 0; i < opts.ntest; ++i ) {
        printf( "m %5d, n %5d, k %5d\n",
                (int) opts.msize[i], (int) opts.nsize[i], (int) opts.ksize[i] );
    }
    printf( "\n" );
    
    printf( "ntest    %d\n", (int) opts.ntest );
    printf( "mmax     %d\n", (int) opts.mmax  );
    printf( "nmax     %d\n", (int) opts.nmax  );
    printf( "kmax     %d\n", (int) opts.kmax  );
    printf( "\n" );
    
    printf( "nb       %d\n", (int) opts.nb       ); 
    printf( "nrhs     %d\n", (int) opts.nrhs     );
    printf( "nstream  %d\n", (int) opts.nstream  );
    printf( "ngpu     %d\n", (int) opts.ngpu     );
    printf( "niter    %d\n", (int) opts.niter    );
    printf( "nthread  %d\n", (int) opts.nthread  );
    printf( "itype    %d\n", (int) opts.itype    );
    printf( "svd_work %d\n", (int) opts.svd_work );
    printf( "\n" );
    
    printf( "check    %s\n", (opts.check  ? "true" : "false") );
    printf( "lapack   %s\n", (opts.lapack ? "true" : "false") );
    printf( "warmup   %s\n", (opts.warmup ? "true" : "false") );
    printf( "all      %s\n", (opts.all    ? "true" : "false") );
    printf( "\n" );
    
    printf( "uplo     %c\n", opts.uplo   );
    printf( "transA   %c\n", opts.transA );
    printf( "transB   %c\n", opts.transB );
    printf( "side     %c\n", opts.side   );
    printf( "diag     %c\n", opts.diag   );
    printf( "jobu     %c\n", opts.jobu   );
    printf( "jobvt    %c\n", opts.jobvt  );
    printf( "jobz     %c\n", opts.jobz   );
    printf( "jobvr    %c\n", opts.jobvr  );
    printf( "jobvl    %c\n", opts.jobvl  );
    
    return 0;
}
