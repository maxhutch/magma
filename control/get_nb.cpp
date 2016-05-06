/*
    -- MAGMA (version 2.0.2) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date May 2016
       
       @author Stan Tomov
       @author Mark Gates
       @author Azzam Haidar
*/

#include "magma_internal.h"

#ifdef __cplusplus
extern "C" {
#endif

// ==== Definition of blocking sizes for Nvidia cards
#ifdef HAVE_CUBLAS

// Optimal block sizes vary with GPU and, to lesser extent, CPU.
// Kepler tuning was on K20c   705 MHz with SandyBridge 2.6 GHz host (bunsen).
// Fermi  tuning was on S2050 1147 MHz with AMD Opteron 2.4 GHz host (romulus).

// Unused arguments are commented out, like /*m*/, to silence compiler warnings,
// especially on Windows.
// Unreachable code should also be removed or commented out.

/* ////////////////////////////////////////////////////////////////////////////
   -- Return nb for potrf based on n
*/
magma_int_t magma_get_spotrf_nb( magma_int_t n )
{
    magma_int_t nb;
    magma_int_t arch = magma_getdevice_arch();
    if ( arch >= 300 ) {       // 3.x Kepler
        if      (n <  1500) nb = 256;
        else                nb = 512;
    }
    else if ( arch >= 200 ) {  // 2.x Fermi
        if      (n <  2048) nb = 256;
        else                nb = 512;
    }
    else {                     // 1.x
        if      (n <  3328) nb = 128;
        else if (n <  4256) nb = 224;
        else                nb = 288;
    }
    return nb;
}

magma_int_t magma_get_dpotrf_nb( magma_int_t n )
{
    magma_int_t nb;
    magma_int_t arch = magma_getdevice_arch();
    if ( arch >= 300 ) {       // 3.x Kepler
        if      (n <  3072) nb = 256;
        else                nb = 512;
    }
    else if ( arch >= 200 ) {  // 2.x Fermi
        nb = 256;
    }
    else {                     // 1.x
        if      (n <  3328) nb = 128;
        else if (n <  4256) nb = 128;
        else                nb = 256;
    }
    return nb;
}

magma_int_t magma_get_cpotrf_nb( magma_int_t n )
{
    magma_int_t nb;
    magma_int_t arch = magma_getdevice_arch();
    if ( arch >= 300 ) {       // 3.x Kepler
        nb = 256;
    }
    else if ( arch >= 200 ) {  // 2.x Fermi
        if      (n <  1500) nb = 192;
        else                nb = 256;
    }
    else {                     // 1.x
        nb = 64;
    }
    return nb;
}

magma_int_t magma_get_zpotrf_nb( magma_int_t n )
{
    magma_int_t nb;
    magma_int_t arch = magma_getdevice_arch();
    if ( arch >= 300 ) {       // 3.x Kepler
        nb = 256;
    }
    else if ( arch >= 200 ) {  // 2.x Fermi
        if      (n <  1500) nb = 192;
        else                nb = 256;
    }
    else {                     // 1.x
        nb = 64;
    }
    return nb;
}

/* ///////////////////////////////////////////////////////////////////////// */

magma_int_t magma_get_zpotrf_right_nb( magma_int_t n )
{
    return 128;
}

magma_int_t magma_get_cpotrf_right_nb( magma_int_t n )
{
    return 128;
}

magma_int_t magma_get_dpotrf_right_nb( magma_int_t n )
{
    return 320;
}

magma_int_t magma_get_spotrf_right_nb( magma_int_t n )
{
    return 128;
}

/* ////////////////////////////////////////////////////////////////////////////
   -- Return nb for geqp3 based on m
*/
magma_int_t magma_get_sgeqp3_nb( magma_int_t /*m*/, magma_int_t /*n*/ )
{
    return 32;
}

magma_int_t magma_get_dgeqp3_nb( magma_int_t /*m*/, magma_int_t /*n*/ )
{
    return 32;
}

magma_int_t magma_get_cgeqp3_nb( magma_int_t /*m*/, magma_int_t /*n*/ )
{
    return 32;
}

magma_int_t magma_get_zgeqp3_nb( magma_int_t /*m*/, magma_int_t /*n*/ )
{
    return 32;
}

/* ////////////////////////////////////////////////////////////////////////////
   -- Return nb for geqrf based on m
*/
magma_int_t magma_get_sgeqrf_nb( magma_int_t m, magma_int_t n )
{
    magma_int_t nb;
    magma_int_t minmn = min( m, n );
    magma_int_t arch = magma_getdevice_arch();
    if ( arch >= 300 ) {       // 3.x Kepler
        if      (minmn <  4096) nb = 96;
        else if (minmn <  7168) nb = 128;
        else if (minmn < 18432) nb = 256;
        else                    nb = 512;
    }
    else if ( arch >= 200 ) {  // 2.x Fermi
        if      (minmn <  3072) nb = 64;
        else if (minmn <  8192) nb = 128;
        else                    nb = 256;
    }
    else {                     // 1.x
        if      (minmn <  2048) nb = 32;
        else if (minmn <  4096) nb = 64;
        else                    nb = 128;
    }
    return nb;
}

magma_int_t magma_get_dgeqrf_nb( magma_int_t m, magma_int_t n )
{
    magma_int_t nb;
    magma_int_t minmn = min( m, n );
    magma_int_t arch = magma_getdevice_arch();
    if ( arch >= 300 ) {       // 3.x Kepler
        if      (minmn <  3072) nb = 64;
        else if (minmn < 10240) nb = 128;
        else                    nb = 256;
    }
    else {                     // 1.x and 2.x Fermi
        if      (minmn <  4096) nb = 64;
        else                    nb = 128;
    }
    return nb;
}

magma_int_t magma_get_cgeqrf_nb( magma_int_t m, magma_int_t n )
{
    magma_int_t nb;
    magma_int_t minmn = min( m, n );
    magma_int_t arch = magma_getdevice_arch();
    if ( arch >= 300 ) {       // 3.x Kepler
        if      (minmn <  4096) nb = 64;
        else                    nb = 128;
    }
    else {                     // 1.x and 2.x Fermi
        if      (minmn <  2048) nb = 32;
        else if (minmn <  4096) nb = 64;
        else                    nb = 128;
    }
    return nb;
}

magma_int_t magma_get_zgeqrf_nb( magma_int_t m, magma_int_t n )
{
    magma_int_t nb;
    magma_int_t minmn = min( m, n );
    magma_int_t arch = magma_getdevice_arch();
    if ( arch >= 300 ) {       // 3.x Kepler
        if      (minmn <  4096) nb = 64;
        else                    nb = 128;
    }
    else if ( arch >= 200 ) {  // 2.x Fermi
        if      (minmn <  2048) nb = 32;
        else if (minmn <  4096) nb = 64;
        else                    nb = 128;
    }
    else {                     // 1.x
        if      (minmn <  1024) nb = 64;
        else                    nb = 128;
    }
    return nb;
}

/* ////////////////////////////////////////////////////////////////////////////
   -- Return nb for geqlf based on m
*/
magma_int_t magma_get_sgeqlf_nb( magma_int_t m, magma_int_t n )
{
    magma_int_t nb;
    magma_int_t minmn = min( m, n );
    magma_int_t arch = magma_getdevice_arch();
    if ( arch >= 200 ) {       // 2.x Fermi
        nb = magma_get_sgeqrf_nb( m, n );
    }
    else {                     // 1.x
        if      (minmn <  1024) nb = 32;
        else if (minmn <  4032) nb = 64;
        else                    nb = 128;
    }
    return nb;
}

magma_int_t magma_get_dgeqlf_nb( magma_int_t m, magma_int_t n )
{
    magma_int_t nb;
    magma_int_t minmn = min( m, n );
    magma_int_t arch = magma_getdevice_arch();
    if ( arch >= 200 ) {       // 2.x Fermi
        nb = magma_get_dgeqrf_nb( m, n );
    }
    else {                     // 1.x
        if      (minmn <  1024) nb = 32;
        else if (minmn <  4032) nb = 64;
        else                    nb = 128;
    }
    return nb;
}

magma_int_t magma_get_cgeqlf_nb( magma_int_t m, magma_int_t n )
{
    magma_int_t nb;
    magma_int_t minmn = min( m, n );
    if      (minmn <  2048) nb = 32;
    else if (minmn <  4032) nb = 64;
    else                    nb = 128;
    return nb;
}

magma_int_t magma_get_zgeqlf_nb( magma_int_t m, magma_int_t n )
{
    magma_int_t nb;
    magma_int_t minmn = min( m, n );
    if      (minmn <  1024) nb = 64;
    else                    nb = 128;
    return nb;
}

/* ////////////////////////////////////////////////////////////////////////////
   -- Nb = nb for gelqf based on m
*/
magma_int_t magma_get_sgelqf_nb( magma_int_t m, magma_int_t n )
{
    return magma_get_sgeqrf_nb( m, n );
}

magma_int_t magma_get_dgelqf_nb( magma_int_t m, magma_int_t n )
{
    magma_int_t nb;
    magma_int_t minmn = min( m, n );
    magma_int_t arch = magma_getdevice_arch();
    if ( arch >= 200 ) {       // 2.x Fermi
        nb = magma_get_dgeqrf_nb( m, n );
    }
    else {                     // 1.x
        if      (minmn <  2048) nb = 32;
        else if (minmn <  4032) nb = 64;
        else                    nb = 128;
    }
    return nb;
}

magma_int_t magma_get_cgelqf_nb( magma_int_t m, magma_int_t n )
{
    magma_int_t nb;
    magma_int_t minmn = min( m, n );
    if      (minmn <  2048) nb = 32;
    else if (minmn <  4032) nb = 64;
    else                    nb = 128;
    return nb;
}

magma_int_t magma_get_zgelqf_nb( magma_int_t m, magma_int_t n )
{
    magma_int_t nb;
    magma_int_t minmn = min( m, n );
    if      (minmn <  1024) nb = 64;
    else                    nb = 128;
    return nb;
}

/* ////////////////////////////////////////////////////////////////////////////
   -- Return nb for getrf based on m
*/
magma_int_t magma_get_sgetrf_nb( magma_int_t m, magma_int_t n )
{
    magma_int_t nb;
    magma_int_t minmn = min( m, n );
    magma_int_t arch = magma_getdevice_arch();
    if ( arch >= 300 ) {       // 3.x Kepler
        if      (minmn <  4096) nb = 256;
        else if (minmn < 18432) nb = 512;
        else                    nb = 1024;
    }
    else if ( arch >= 200 ) {  // 2.x Fermi
        if      (minmn <  3072) nb = 128;
        else if (minmn < 10240) nb = 256;
        else                    nb = 512;
    }
    else {                     // 1.x
        if      (minmn <  2048) nb = 64;
        else                    nb = 128;
    }
    return nb;
}

magma_int_t magma_get_dgetrf_nb( magma_int_t m, magma_int_t n )
{
    magma_int_t nb;
    magma_int_t minmn = min( m, n );
    magma_int_t arch = magma_getdevice_arch();
    if ( arch >= 300 ) {       // 3.x Kepler
        if      (minmn <  3072) nb = 128;
        else if (minmn <  8192) nb = 256;
        else                    nb = 512;
    }
    else if ( arch >= 200 ) {  // 2.x Fermi
        if      (minmn <  3072) nb = 128;
        else if (minmn < 10240) nb = 256;
        else                    nb = 512;
    }
    else {                     // 1.x
        if      (minmn <  2048) nb = 64;
        else                    nb = 128;
    }
    return nb;
}

magma_int_t magma_get_cgetrf_nb( magma_int_t m, magma_int_t n )
{
    magma_int_t nb;
    magma_int_t minmn = min( m, n );
    magma_int_t arch = magma_getdevice_arch();
    if ( arch >= 300 ) {       // 3.x Kepler
        if      (minmn < 4096) nb = 64;
        else if (minmn < 8192) nb = 256;
        else                   nb = 512;
    }
    else if ( arch >= 200 ) {  // 2.x Fermi
        if      (minmn <  2048) nb = 64;
        else                    nb = 128;
    }
    else {                     // 1.x
        if      (minmn <  2048) nb = 64;
        else                    nb = 128;
    }
    return nb;
}

magma_int_t magma_get_zgetrf_nb( magma_int_t m, magma_int_t n )
{
    magma_int_t nb;
    magma_int_t minmn = min( m, n );
    magma_int_t arch = magma_getdevice_arch();
    if ( arch >= 300 ) {       // 3.x Kepler
        if      (minmn < 4096) nb = 64;
        else if (minmn < 8192) nb = 256;
        else                   nb = 512;
    }
    else if ( arch >= 200 ) {  // 2.x Fermi
        if      (minmn < 4096) nb = 64;
        else                   nb = 128;
    }
    else {                     // 1.x
        nb = 128;
    }
    return nb;
}

/* ////////////////////////////////////////////////////////////////////////////
   -- Return nb for gehrd based on n
*/
magma_int_t magma_get_sgehrd_nb( magma_int_t n )
{
    magma_int_t nb;
    magma_int_t arch = magma_getdevice_arch();
    if ( arch >= 200 ) {       // 2.x Fermi
        if      (n <  1024) nb = 32;
        else                nb = 96;
    }
    else {                     // 1.x
        if      (n <  1024) nb = 32;
        else                nb = 64;
    }
    return nb;
}

magma_int_t magma_get_dgehrd_nb( magma_int_t n )
{
    magma_int_t nb;
    if      (n <  2048) nb = 32;
    else                nb = 64;
    return nb;
}

magma_int_t magma_get_cgehrd_nb( magma_int_t n )
{
    magma_int_t nb;
    if      (n <  1024) nb = 32;
    else                nb = 64;
    return nb;
}

magma_int_t magma_get_zgehrd_nb( magma_int_t n )
{
    magma_int_t nb;
    if      (n <  2048) nb = 32;
    else                nb = 64;
    return nb;
}

/* ////////////////////////////////////////////////////////////////////////////
   -- Return nb for sytrd based on n
      Must be 64 due to zhemv_mgpu restrictions.
*/
magma_int_t magma_get_ssytrd_nb( magma_int_t /*n*/ )
{
    return 64;
}

magma_int_t magma_get_dsytrd_nb( magma_int_t /*n*/ )
{
    return 64;
}

magma_int_t magma_get_chetrd_nb( magma_int_t /*n*/ )
{
    return 64;
}

magma_int_t magma_get_zhetrd_nb( magma_int_t /*n*/ )
{
    return 64;
}

/* ////////////////////////////////////////////////////////////////////////////
   -- Return nb for sytrf based on n
*/
magma_int_t magma_get_zhetrf_nb( magma_int_t n )
{
    return 256;
}

magma_int_t magma_get_chetrf_nb( magma_int_t n )
{
    return 256;
}

magma_int_t magma_get_dsytrf_nb( magma_int_t n )
{
    return 96;
}

magma_int_t magma_get_ssytrf_nb( magma_int_t n )
{
    return 256;
}

/* //////////////////////////////////////////////////////////////////////// */
magma_int_t magma_get_zhetrf_aasen_nb( magma_int_t n )
{
    return 256;
}

magma_int_t magma_get_chetrf_aasen_nb( magma_int_t n )
{
    return 256;
}

magma_int_t magma_get_dsytrf_aasen_nb( magma_int_t n )
{
    return 256;
}

magma_int_t magma_get_ssytrf_aasen_nb( magma_int_t n )
{
    return 256;
}

/* //////////////////////////////////////////////////////////////////////// */
magma_int_t magma_get_zhetrf_nopiv_nb( magma_int_t n )
{
    return 320;
}

magma_int_t magma_get_chetrf_nopiv_nb( magma_int_t n )
{
    return 96;
}

magma_int_t magma_get_dsytrf_nopiv_nb( magma_int_t n )
{
    return 320;
}

magma_int_t magma_get_ssytrf_nopiv_nb( magma_int_t n )
{
    return 320;
}

/* ////////////////////////////////////////////////////////////////////////////
   -- Return nb for gebrd based on m
*/
magma_int_t magma_get_sgebrd_nb( magma_int_t /*m*/, magma_int_t /*n*/ )
{
    return 32;
}

magma_int_t magma_get_dgebrd_nb( magma_int_t /*m*/, magma_int_t /*n*/ )
{
    return 32;
}

magma_int_t magma_get_cgebrd_nb( magma_int_t /*m*/, magma_int_t /*n*/ )
{
    return 32;
}

magma_int_t magma_get_zgebrd_nb( magma_int_t /*m*/, magma_int_t /*n*/ )
{
    return 32;
}

/* ////////////////////////////////////////////////////////////////////////////
   -- Return nb for sygst based on n
*/
magma_int_t magma_get_ssygst_nb( magma_int_t n )
{
    magma_int_t nb;
    magma_int_t arch = magma_getdevice_arch();
    if ( arch >= 300 ) {       // 3.x Kepler
        if      (n <  4096) nb = 768;
        else                nb = 1536;
    }
    else if ( arch >= 200 ) {  // 2.x Fermi
        if      (n <  2048) nb = 512;
        else                nb = 1024;
    }
    else {                     // 1.x
        nb = 64;
    }
    return nb;
}

magma_int_t magma_get_dsygst_nb( magma_int_t n )
{
    magma_int_t nb;
    magma_int_t arch = magma_getdevice_arch();
    if ( arch >= 300 ) {       // 3.x Kepler
        if      (n <  2048) nb = 384;
        else                nb = 768;
    }
    else if ( arch >= 200 ) {  // 2.x Fermi
        nb = 512;
    }
    else {                     // 1.x
        nb = 64;
    }
    return nb;
}

magma_int_t magma_get_chegst_nb( magma_int_t n )
{
    magma_int_t nb;
    magma_int_t arch = magma_getdevice_arch();
    if ( arch >= 300 ) {       // 3.x Kepler
        if      (n <  2048) nb = 384;
        else                nb = 768;
    }
    else if ( arch >= 200 ) {  // 2.x Fermi
        nb = 512;
    }
    else {                     // 1.x
        nb = 64;
    }
    return nb;
}

magma_int_t magma_get_zhegst_nb( magma_int_t /*n*/ )
{
    magma_int_t nb;
    magma_int_t arch = magma_getdevice_arch();
    if ( arch >= 300 ) {       // 3.x Kepler
        nb = 384;
    }
    else if ( arch >= 200 ) {  // 2.x Fermi
        nb = 256;
    }
    else {                     // 1.x
        nb = 64;
    }
    return nb;
}

/* ////////////////////////////////////////////////////////////////////////////
   -- Return nb for getri based on n
*/
magma_int_t magma_get_sgetri_nb( magma_int_t /*n*/ )
{
    return 64;
}

magma_int_t magma_get_dgetri_nb( magma_int_t /*n*/ )
{
    return 64;
}

magma_int_t magma_get_cgetri_nb( magma_int_t /*n*/ )
{
    return 64;
}

magma_int_t magma_get_zgetri_nb( magma_int_t /*n*/ )
{
    return 64;
}

/* ////////////////////////////////////////////////////////////////////////////
   -- Return nb for gesvd based on m, n
*/
magma_int_t magma_get_sgesvd_nb( magma_int_t m, magma_int_t n )
{
    return magma_get_sgebrd_nb( m, n );
}

magma_int_t magma_get_dgesvd_nb( magma_int_t m, magma_int_t n )
{
    return magma_get_dgebrd_nb( m, n );
}

magma_int_t magma_get_cgesvd_nb( magma_int_t m, magma_int_t n )
{
    return magma_get_cgebrd_nb( m, n );
}

magma_int_t magma_get_zgesvd_nb( magma_int_t m, magma_int_t n )
{
    return magma_get_zgebrd_nb( m, n );
}

/* ////////////////////////////////////////////////////////////////////////////
   -- Return nb for sygst based on n
*/
magma_int_t magma_get_ssygst_nb_m( magma_int_t /*n*/ )
{
    return 256; //to be updated

    /*
    magma_int_t nb;
    magma_int_t arch = magma_getdevice_arch();
    if ( arch >= 300 ) {       // 3.x Kepler
        if      (n <  4096) nb = 768;
        else                nb = 1536;
    }
    else if ( arch >= 200 ) {  // 2.x Fermi
        if      (n <  2048) nb = 512;
        else                nb = 1024;
    }
    else {                     // 1.x
        nb = 64;
    }
    return nb;
    */
}

magma_int_t magma_get_dsygst_nb_m( magma_int_t /*n*/ )
{
    return 256; //to be updated

    /*
    magma_int_t nb;
    magma_int_t arch = magma_getdevice_arch();
    if ( arch >= 300 ) {       // 3.x Kepler
        if      (n <  2048) nb = 384;
        else                nb = 768;
    }
    else if ( arch >= 200 ) {  // 2.x Fermi
        nb = 512;
    }
    else {                     // 1.x
        nb = 64;
    }
    return nb;
    */
}

magma_int_t magma_get_chegst_nb_m( magma_int_t /*n*/ )
{
    return 256; //to be updated

    /*
    magma_int_t nb;
    magma_int_t arch = magma_getdevice_arch();
    if ( arch >= 300 ) {       // 3.x Kepler
        if      (n <  2048) nb = 384;
        else                nb = 768;
    }
    else if ( arch >= 200 ) {  // 2.x Fermi
        nb = 512;
    }
    else {                     // 1.x
        nb = 64;
    }
    return nb;
    */
}

magma_int_t magma_get_zhegst_nb_m( magma_int_t /*n*/ )
{
    return 256; //to be updated

    /*
    magma_int_t nb;
    magma_int_t arch = magma_getdevice_arch();
    if ( arch >= 300 ) {       // 3.x Kepler
        nb = 384;
    }
    else if ( arch >= 200 ) {  // 2.x Fermi
        nb = 256;
    }
    else {                     // 1.x
        nb = 64;
    }
    return nb;
    */
}


    /////////////////////////////////////////
    /////////////////////////////////////////
    // Parameters for 2-stage eigensolvers //
    /////////////////////////////////////////
    /////////////////////////////////////////

/* ////////////////////////////////////////////////////////////////////////////
   -- Return gpu_cpu_perf for  2 stage TRD
*/
magma_int_t magma_get_sbulge_gcperf( )
{
    magma_int_t nb;
    magma_int_t arch = magma_getdevice_arch();
    if ( arch >= 300 ) {       // 3.x Kepler + SB
        nb = 37;
    }
    else if ( arch >= 200 ) {  // 2.x Fermi
        nb = 15000;
    }
    else {                     // 1.x
        nb = 10000;
    }
    return nb;
}

magma_int_t magma_get_dbulge_gcperf( )
{
    magma_int_t nb;
    magma_int_t arch = magma_getdevice_arch();
    if ( arch >= 300 ) {       // 3.x Kepler + SB
        nb = 37;
    }
    else if ( arch >= 200 ) {  // 2.x Fermi
        nb = 15000;
    }
    else {                     // 1.x
        nb = 10000;
    }
    return nb;
}

magma_int_t magma_get_cbulge_gcperf( )
{
    magma_int_t nb;
    magma_int_t arch = magma_getdevice_arch();
    if ( arch >= 300 ) {       // 3.x Kepler + SB
        nb = 50;
    }
    else if ( arch >= 200 ) {  // 2.x Fermi
        nb = 15000;
    }
    else {                     // 1.x
        nb = 10000;
    }
    return nb;
}

magma_int_t magma_get_zbulge_gcperf( )
{
    magma_int_t nb;
    magma_int_t arch = magma_getdevice_arch();
    if ( arch >= 300 ) {       // 3.x Kepler + SB
        nb = 50;
    }
    else if ( arch >= 200 ) {  // 2.x Fermi
        nb = 15000;
    }
    else {                     // 1.x
        nb = 10000;
    }
    return nb;
}


/* ////////////////////////////////////////////////////////////////////////////
   -- Return smlsiz for the divide and conquewr routine dlaex0 dstedx zstedx
*/
magma_int_t magma_get_smlsize_divideconquer()
{
    return 128;
}



/* ////////////////////////////////////////////////////////////////////////////
   -- Return nb for  2 stage TRD
*/
magma_int_t magma_get_sbulge_nb( magma_int_t /*n*/, magma_int_t /*nbthreads*/  )
{
    magma_int_t nb;
    magma_int_t arch = magma_getdevice_arch();
    if ( arch >= 300 ) {       // 3.x Kepler + SB
        nb = 128;
    }
    else if ( arch >= 200 ) {  // 2.x Fermi
        nb = 64;
    }
    else {                     // 1.x
        nb = 64;
    }
    return nb;
}

magma_int_t magma_get_dbulge_nb( magma_int_t /*n*/, magma_int_t /*nbthreads*/  )
{
    magma_int_t nb;
    magma_int_t arch = magma_getdevice_arch();
    if ( arch >= 300 ) {       // 3.x Kepler + SB
        nb = 128;
    }
    else if ( arch >= 200 ) {  // 2.x Fermi
        nb = 128;
    }
    else {                     // 1.x
        nb = 64;
    }
    return nb;
}

magma_int_t magma_get_cbulge_nb( magma_int_t /*n*/, magma_int_t nbthreads  )
{
    magma_int_t nb;
    magma_int_t arch = magma_getdevice_arch();
    if ( arch >= 300 ) {       // 3.x Kepler + SB
        if ( nbthreads > 14 )
            nb = 128;
        else
            nb = 64;
    }
    else if ( arch >= 200 ) {  // 2.x Fermi
        nb = 64;
    }
    else {                     // 1.x
        nb = 64;
    }
    return nb;
}

magma_int_t magma_get_zbulge_nb( magma_int_t /*n*/, magma_int_t nbthreads )
{
    magma_int_t nb;
    magma_int_t arch = magma_getdevice_arch();
    if ( arch >= 300 ) {       // 3.x Kepler + SB
        if ( nbthreads > 14 )
            nb = 128;
        else
            nb = 64;
    }
    else if ( arch >= 200 ) {  // 2.x Fermi
        nb = 64;
    }
    else {                     // 1.x
        nb = 64;
    }
    return nb;
}


/* ////////////////////////////////////////////////////////////////////////////
   -- Return Vblksiz for  2 stage TRD
*/
magma_int_t magma_get_sbulge_vblksiz( magma_int_t /*n*/, magma_int_t nb, magma_int_t /*nbthreads*/  )
{
    magma_int_t size;
    magma_int_t arch = magma_getdevice_arch();
    if ( arch >= 300 ) {       // 3.x Kepler + SB
        size = min(nb, 128);
    }
    else {                     // 2.x Fermi or 1.x
        size = min(nb, 64);
    }
    return size;
}

magma_int_t magma_get_dbulge_vblksiz( magma_int_t /*n*/, magma_int_t nb, magma_int_t /*nbthreads*/  )
{
    magma_int_t size;
    magma_int_t arch = magma_getdevice_arch();
    if ( arch >= 300 ) {       // 3.x Kepler + SB
        size = min(nb, 64);
    }
    else {                     // 2.x Fermi or 1.x
        size = min(nb, 48);
    }
    return size;
}

magma_int_t magma_get_cbulge_vblksiz( magma_int_t /*n*/, magma_int_t nb, magma_int_t nbthreads )
{
    magma_int_t size;
    magma_int_t arch = magma_getdevice_arch();
    if ( arch >= 300 ) {       // 3.x Kepler + SB
        if ( nbthreads > 14 )
            size = min(nb, 48);
        else
            size = min(nb, 48);
    }
    else {                     // 2.x Fermi or 1.x
        size = min(nb, 48);
    }
    return size;
}

magma_int_t magma_get_zbulge_vblksiz( magma_int_t /*n*/, magma_int_t nb, magma_int_t nbthreads )
{
    magma_int_t size;
    magma_int_t arch = magma_getdevice_arch();
    if ( arch >= 300 ) {       // 3.x Kepler + SB
        if ( nbthreads > 14 )
            size = min(nb, 64);
        else
            size = min(nb, 32);
    }
    else {                     // 2.x Fermi or 1.x
        size = min(nb, 48);
    }
    return size;
}


/* ////////////////////////////////////////////////////////////////////////////
   -- Return nb for  2 stage TRD_MGPU
*/
magma_int_t magma_get_sbulge_mgpu_nb( magma_int_t /*n*/ )
{
    magma_int_t nb;
    magma_int_t arch = magma_getdevice_arch();
    if ( arch >= 300 ) {       // 3.x Kepler + SB
        nb = 128;
    }
    else if ( arch >= 200 ) {  // 2.x Fermi
        nb = 64;
    }
    else {                     // 1.x
        nb = 64;
    }
    return nb;
}

magma_int_t magma_get_dbulge_mgpu_nb( magma_int_t /*n*/ )
{
    magma_int_t nb;
    magma_int_t arch = magma_getdevice_arch();
    if ( arch >= 300 ) {       // 3.x Kepler + SB
        nb = 128;
    }
    else if ( arch >= 200 ) {  // 2.x Fermi
        nb = 64;
    }
    else {                     // 1.x
        nb = 64;
    }
    return nb;
}

magma_int_t magma_get_cbulge_mgpu_nb( magma_int_t /*n*/ )
{
    magma_int_t nb;
    magma_int_t arch = magma_getdevice_arch();
    if ( arch >= 300 ) {       // 3.x Kepler + SB
        nb = 64;
    }
    else if ( arch >= 200 ) {  // 2.x Fermi
        nb = 64;
    }
    else {                     // 1.x
        nb = 64;
    }
    return nb;
}

magma_int_t magma_get_zbulge_mgpu_nb( magma_int_t /*n*/ )
{
    magma_int_t nb;
    magma_int_t arch = magma_getdevice_arch();
    if ( arch >= 300 ) {       // 3.x Kepler + SB
        nb = 64;
    }
    else if ( arch >= 200 ) {  // 2.x Fermi
        nb = 64;
    }
    else {                     // 1.x
        nb = 64;
    }
    return nb;
}

#endif  // HAVE_CUBLAS

#ifdef __cplusplus
} // extern "C"
#endif
