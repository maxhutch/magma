/*
    -- MAGMA (version 1.6.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date January 2015

       @precisions normal z -> s d c
       @author Azzam Haidar
       
       Work in progress.
*/
#include "common_magma.h"

extern "C"
magma_int_t magma_buildconnection_mgpu(  magma_int_t gnode[MagmaMaxGPUs+2][MagmaMaxGPUs+2], magma_int_t *nbcmplx, magma_int_t ngpu)
{
    magma_int_t *deviceid = (magma_int_t *) malloc(ngpu*sizeof(magma_int_t));
    memset(deviceid, 0, ngpu*sizeof(magma_int_t));

    nbcmplx[0] =0;


    //printf(" Initializing....\n\n");
    //printf(" This machine has %d GPU\n", ngpu);

    //printf(" cudaSuccess %d, cudaErrorInvalidDevice %d, cudaErrorPeerAccessAlreadyEnabled %d, cudaErrorInvalidValue %d \n",
    //       cudaSuccess, cudaErrorInvalidDevice, cudaErrorPeerAccessAlreadyEnabled, cudaErrorInvalidValue );

    int samecomplex=-1;
    cudaError_t err;
    cudaDeviceProp prop;

    magma_int_t cmplxnb = 0;
    magma_int_t cmplxid = 0;
    magma_int_t lcgpunb = 0;
    for( magma_int_t d = 0; d < ngpu; ++d ) {
        // check for unified memory & enable peer memory access between all GPUs.            
        magma_setdevice( d );
        cudaGetDeviceProperties( &prop, d );
        if ( ! prop.unifiedAddressing ) {
            printf( "device %d doesn't support unified addressing\n", (int) d );
            free(deviceid);
            return -1;
        }
        // add this device to the list if not added yet.
        // not added yet meaning belong to a new complex
        if(deviceid[d]==0){
            cmplxnb           = cmplxnb+1;
            cmplxid           = cmplxnb-1;
            gnode[cmplxid][MagmaMaxGPUs] = 1;
            lcgpunb           = gnode[cmplxid][MagmaMaxGPUs]-1;
            gnode[cmplxid][lcgpunb] = d;
            deviceid[d]=-1;
        }
        //printf("DEVICE %d : \n", d);

        for( magma_int_t d2 = d+1; d2 < ngpu; ++d2 ) {
            // check for unified memory & enable peer memory access between all GPUs.            
            magma_setdevice( d2 );
            cudaGetDeviceProperties( &prop, d2 );
            if ( ! prop.unifiedAddressing ) {
                printf( "device %d doesn't support unified addressing\n", (int) d2 );
                free(deviceid);
                return -1;
            }

            /* TODO err = */ cudaDeviceCanAccessPeer(&samecomplex, d, d2);

            //printf(" device %d and device %d have samecomplex= %d\n", d, d2, samecomplex);
            if(samecomplex==1){
                // d and d2 are on the same complex so add them, note that d is already added
                // so just enable the peer Access for d and enable+add d2.
                // FOR d:
                magma_setdevice( d );
                err   = cudaDeviceEnablePeerAccess( d2, 0 );
                //printf("enabling devide %d ==> %d  error %d\n", d, d2, err);
                if ( err != cudaSuccess && err != cudaErrorPeerAccessAlreadyEnabled ) {
                    printf( "device %d cudaDeviceEnablePeerAccess error %d\n", (int) d2, (int) err );
                    free(deviceid);
                    return -2;
                }

                // FOR d2:
                magma_setdevice( d2 );
                err   = cudaDeviceEnablePeerAccess( d, 0 );
                //printf("enabling devide %d ==> %d  error %d\n", d2, d, err);
                if((err==cudaSuccess)||(err==cudaErrorPeerAccessAlreadyEnabled)){
                    if(deviceid[d2]==0){
                        //printf("adding device %d\n", d2);
                        gnode[cmplxid][MagmaMaxGPUs] = gnode[cmplxid][MagmaMaxGPUs]+1;
                        lcgpunb           = gnode[cmplxid][MagmaMaxGPUs]-1;
                        gnode[cmplxid][lcgpunb] = d2;
                        deviceid[d2]=-1;
                    }
                }else{
                    printf( "device %d cudaDeviceEnablePeerAccess error %d\n", (int) d, (int) err );
                    free(deviceid);
                    return -2;
                }
            }
        }
    }

    nbcmplx[0] = cmplxnb;
    free(deviceid); 
    return cmplxnb;
}
