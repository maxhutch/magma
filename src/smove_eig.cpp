/*
    -- MAGMA (version 1.4.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       December 2013

       @author Raffaele Solca
       @author Azzam Haidar

       @generated s Tue Dec 17 13:18:36 2013

*/
#include "common_magma.h"
extern"C" void 
magma_smove_eig(char range, magma_int_t n, float *w, magma_int_t *il,
                     magma_int_t *iu, float vl, float vu, magma_int_t *m)
{
    char range_[2] = {range, 0};

    magma_int_t valeig, indeig, i;

    valeig = lapackf77_lsame( range_, "V" );
    indeig = lapackf77_lsame( range_, "I" );

    if (indeig){
        *m = *iu - *il + 1;
        if(*il > 1)
            for (i = 0; i < *m; ++i)
                w[i] = w[*il - 1 + i];
    }
    else if(valeig){
        *il=1;
        *iu=n;
        for (i = 0; i < n; ++i){
            if (w[i] > vu){
                *iu = i;
                break;
            }
            else if (w[i] < vl)
                ++*il;
            else if (*il > 1)
                w[i-*il+1]=w[i];
        }
        *m = *iu - *il + 1;
    }
    else{
        *il = 1;
        *iu = n;
        *m = n;
    }

    return;
}

