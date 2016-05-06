/*
    -- MAGMA (version 2.0.2) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date May 2016

       @author Mark Gates
*/

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include <complex>

// includes, project
#include "magma_v2.h"
#include "magma_lapack.h"
#include "magma_operators.h"
#include "testings.h"


////////////////////////////////////////////////////////////////////////////
// check( flag ) keeps tally in gStatus of tests that fail
magma_int_t gStatus;

void check_( bool flag, const char* msg, int line )
{
    if ( ! flag ) {
        gStatus += 1;
        printf( "line %d: %s failed\n", line, msg );
    }
}

#define check( flag ) check_( flag, #flag, __LINE__ )


////////////////////////////////////////////////////////////////////////////
void test_sqrt( double dtol, double stol )
{
    magma_int_t s = gStatus;
    
    float              sx, sr;
    double             dx, dr;
    magmaFloatComplex  cx, cr;
    magmaDoubleComplex zx, zr;
    
    sx = 4;
    dx = 4;
    cx = MAGMA_C_MAKE( 4, 0 );
    zx = MAGMA_Z_MAKE( 4, 0 );
    
    sr = magma_ssqrt( sx );
    dr = magma_dsqrt( dx );
    cr = magma_csqrt( cx );
    zr = magma_zsqrt( zx );
    
    // check that single, double, single-complex, double-complex agree
    // check that r^2 == x
    // check that r > 0 (principal sqrt)
    check( sr == dr );
    check( sr == cr );
    check( sr == zr );
    check( fabs( sr*sr - sx ) < stol );
    check( fabs( dr*dr - dx ) < dtol );
    check( fabs( cr*cr - cx ) < stol );
    check( fabs( zr*zr - zx ) < dtol );
    check( real(cr) >= 0 );
    
    // complex sqrt
    cx = MAGMA_C_MAKE( -4, 0 );
    zx = MAGMA_Z_MAKE( -4, 0 );
    cr = magma_csqrt( cx );
    zr = magma_zsqrt( zx );
    check( fabs( real(cr) - real(zr) ) < stol &&
           fabs( imag(cr) - imag(zr) ) < stol );
    check( fabs( cr*cr - cx ) < stol );
    check( fabs( zr*zr - zx ) < dtol );
    check( real(cr) >= 0 );
    
    cx = MAGMA_C_MAKE( 2, 2 );
    zx = MAGMA_Z_MAKE( 2, 2 );
    cr = magma_csqrt( cx );
    zr = magma_zsqrt( zx );
    check( fabs( real(cr) - real(zr) ) < stol &&
           fabs( imag(cr) - imag(zr) ) < stol );
    check( fabs( cr*cr - cx ) < stol );
    check( fabs( zr*zr - zx ) < dtol );
    check( real(cr) >= 0 );
    
    cx = MAGMA_C_MAKE( -2, 2 );
    zx = MAGMA_Z_MAKE( -2, 2 );
    cr = magma_csqrt( cx );
    zr = magma_zsqrt( zx );
    check( fabs( real(cr) - real(zr) ) < stol &&
           fabs( imag(cr) - imag(zr) ) < stol );
    check( fabs( cr*cr - cx ) < stol );
    check( fabs( zr*zr - zx ) < dtol );
    check( real(cr) >= 0 );
    
    cx = MAGMA_C_MAKE( 2, -2 );
    zx = MAGMA_Z_MAKE( 2, -2 );
    cr = magma_csqrt( cx );
    zr = magma_zsqrt( zx );
    check( fabs( real(cr) - real(zr) ) < stol &&
           fabs( imag(cr) - imag(zr) ) < stol );
    check( fabs( cr*cr - cx ) < stol );
    check( fabs( zr*zr - zx ) < dtol );
    check( real(cr) >= 0 );
    
    cx = MAGMA_C_MAKE( -2, -2 );
    zx = MAGMA_Z_MAKE( -2, -2 );
    cr = magma_csqrt( cx );
    zr = magma_zsqrt( zx );
    check( fabs( real(cr) - real(zr) ) < stol &&
           fabs( imag(cr) - imag(zr) ) < stol );
    check( fabs( cr*cr - cx ) < stol );
    check( fabs( zr*zr - zx ) < dtol );
    check( real(cr) >= 0 );
    
    printf( "sqrt                            %s\n", (s == gStatus ? "ok" : "failed"));
}


int main( int argc, char** argv)
{
    TESTING_INIT();
    
    gStatus = 0;
    magma_int_t s;

    magmaDoubleComplex za, zb, zc,  za2, zb2, zc2;
    magmaFloatComplex  ca, cb, cc,  ca2, cb2, cc2;
    double             da, db, dc,  da2, db2, dc2;
    float              sa, sb, sc,  sa2, sb2, sc2;
    bool               eq,          eq2;

    std::complex<double> za3, zb3, zc3;
    std::complex<float>  ca3, cb3, cc3;
    double               da3;  //, db3, dc3;
    float                sa3;  //, sb3, sc3;
    bool                 eq3;

    magma_opts opts;
    opts.parse_opts( argc, argv );

    // most operators are simple, so should be exactly the same.
    // some operators (divide, abs) have different implementations, so may differ slightly, within tol.
    double dtol = opts.tolerance * lapackf77_dlamch("E");
    double stol = opts.tolerance * lapackf77_slamch("E");

    // --------------------
    // MAGMA operator                 std::complex operator                      verify
    s = gStatus;
    za = MAGMA_Z_MAKE( 1.23, 2.45 );  za3 = std::complex<double>( 1.23, 2.45 );  check( za == MAGMA_Z_MAKE( real(za3), imag(za3) ) );
    zb = MAGMA_Z_MAKE( 3.14, 2.72 );  zb3 = std::complex<double>( 3.14, 2.72 );  check( zb == MAGMA_Z_MAKE( real(zb3), imag(zb3) ) );
    zc = conj( za );                  zc3 = conj( za3 );                         check( zc == MAGMA_Z_MAKE( real(zc3), imag(zc3) ) );
    zc = -za;                         zc3 = -za3;                                check( zc == MAGMA_Z_MAKE( real(zc3), imag(zc3) ) );
    zc = za + zb;                     zc3 = za3 + zb3;                           check( zc == MAGMA_Z_MAKE( real(zc3), imag(zc3) ) );
    zc = za - zb;                     zc3 = za3 - zb3;                           check( zc == MAGMA_Z_MAKE( real(zc3), imag(zc3) ) );
    zc = za * zb;                     zc3 = za3 * zb3;                           check( fabs( zc - MAGMA_Z_MAKE( real(zc3), imag(zc3) ) ) < dtol );
    zc = za / zb;                     zc3 = za3 / zb3;                           check( fabs( zc - MAGMA_Z_MAKE( real(zc3), imag(zc3) ) ) < dtol );

    da = fabs( za );                  da3 = std::abs( za3 );                     check( da == da3 );
    //da = abs1( za );                da3 = abs1( za3 );                         check( da == da3 );  // no std::abs1
    da = real( za );                  da3 = real( za3 );                         check( da == da3 );
    da = imag( za );                  da3 = imag( za3 );                         check( da == da3 );

    zc = za;                          zc3 = za3;                                 check( zc == MAGMA_Z_MAKE( real(zc3), imag(zc3) ) );
    eq = (za == zb);                  eq3 = (za3 == zb3);                        check( eq == eq3 );
    eq = (za == zc);                  eq3 = (za3 == zc3);                        check( eq == eq3 );
    printf( "std::complex<double> operators  %s\n", (s == gStatus ? "ok" : "failed"));

    // --------------------
    // MAGMA operator                 std::complex operator                      verify
    s = gStatus;
    ca = MAGMA_C_MAKE( 1.23, 2.45 );  ca3 = std::complex<float>( 1.23, 2.45 );   check( ca == MAGMA_C_MAKE( real(ca3), imag(ca3) ) );
    cb = MAGMA_C_MAKE( 3.14, 2.72 );  cb3 = std::complex<float>( 3.14, 2.72 );   check( cb == MAGMA_C_MAKE( real(cb3), imag(cb3) ) );
    cc = conj( ca );                  cc3 = conj( ca3 );                         check( cc == MAGMA_C_MAKE( real(cc3), imag(cc3) ) );
    cc = -ca;                         cc3 = -ca3;                                check( cc == MAGMA_C_MAKE( real(cc3), imag(cc3) ) );
    cc = ca + cb;                     cc3 = ca3 + cb3;                           check( cc == MAGMA_C_MAKE( real(cc3), imag(cc3) ) );
    cc = ca - cb;                     cc3 = ca3 - cb3;                           check( cc == MAGMA_C_MAKE( real(cc3), imag(cc3) ) );
    cc = ca * cb;                     cc3 = ca3 * cb3;                           check( fabs( cc - MAGMA_C_MAKE( real(cc3), imag(cc3) ) ) < stol );
    cc = ca / cb;                     cc3 = ca3 / cb3;                           check( fabs( cc - MAGMA_C_MAKE( real(cc3), imag(cc3) ) ) < stol );

    sa = fabs( ca );                  sa3 = std::abs( ca3 );                     check( fabs( sa - sa3 ) < stol );
    //sa = abs1( ca );                sa3 = abs1( ca3 );                         check( sa == sa3 );  // no std::abs1
    sa = real( ca );                  sa3 = real( ca3 );                         check( sa == sa3 );
    sa = imag( ca );                  sa3 = imag( ca3 );                         check( sa == sa3 );

    cc = ca;                          cc3 = ca3;                                 check( cc == MAGMA_C_MAKE( real(cc3), imag(cc3) ) );
    eq = (ca == cb);                  eq3 = (ca3 == cb3);                        check( eq == eq3 );
    eq = (ca == cc);                  eq3 = (ca3 == cc3);                        check( eq == eq3 );
    printf( "std::complex<float>  operators  %s\n", (s == gStatus ? "ok" : "failed"));

    // --------------------
    // MAGMA operator                 MAGMA MACRO                        verify
    s = gStatus;
    za = MAGMA_Z_MAKE( 1.23, 2.45 );  za2 = MAGMA_Z_MAKE( 1.23, 2.45 );  check( za == za2 );
    zb = MAGMA_Z_MAKE( 3.14, 2.72 );  zb2 = MAGMA_Z_MAKE( 3.14, 2.72 );  check( zb == zb2 );
    zc = conj( za );                  zc2 = MAGMA_Z_CONJ( za2 );         check( zc == zc2 );
    zc = -za;                         zc2 = MAGMA_Z_NEGATE( za2 );       check( zc == zc2 );
    zc = za + zb;                     zc2 = MAGMA_Z_ADD( za2, zb2 );     check( zc == zc2 );
    zc = za - zb;                     zc2 = MAGMA_Z_SUB( za2, zb2 );     check( zc == zc2 );
    zc = za * zb;                     zc2 = MAGMA_Z_MUL( za2, zb2 );     check( zc == zc2 );
    zc = za / zb;                     zc2 = MAGMA_Z_DIV( za2, zb2 );     check( fabs( zc - zc2 ) < dtol );

    da = fabs( za );                  da2 = MAGMA_Z_ABS(  za2 );         check( da == da2 );
    da = abs1( za );                  da2 = MAGMA_Z_ABS1( za2 );         check( da == da2 );
    da = real( za );                  da2 = MAGMA_Z_REAL( za2 );         check( da == da2 );
    da = imag( za );                  da2 = MAGMA_Z_IMAG( za2 );         check( da == da2 );

    zc = za;                          zc2 = za2;                         check( zc == zc2 );
    eq = (za == zb);                  eq2 = MAGMA_Z_EQUAL( za2, zb2 );   check( eq == eq2 );
    eq = (za == zc);                  eq2 = MAGMA_Z_EQUAL( za2, zc2 );   check( eq == eq2 );
    printf( "magmaDoubleComplex operators    %s\n", (s == gStatus ? "ok" : "failed"));

    // --------------------
    // MAGMA operator                 MAGMA MACRO                        verify
    s = gStatus;
    ca = MAGMA_C_MAKE( 1.23, 2.45 );  ca2 = MAGMA_C_MAKE( 1.23, 2.45 );  check( ca == ca2 );
    cb = MAGMA_C_MAKE( 3.14, 2.72 );  cb2 = MAGMA_C_MAKE( 3.14, 2.72 );  check( cb == cb2 );
    cc = conj( ca );                  cc2 = MAGMA_C_CONJ( ca2 );         check( cc == cc2 );
    cc = -ca;                         cc2 = MAGMA_C_NEGATE( ca2 );       check( cc == cc2 );
    cc = ca + cb;                     cc2 = MAGMA_C_ADD( ca2, cb2 );     check( cc == cc2 );
    cc = ca - cb;                     cc2 = MAGMA_C_SUB( ca2, cb2 );     check( cc == cc2 );
    cc = ca * cb;                     cc2 = MAGMA_C_MUL( ca2, cb2 );     check( cc == cc2 );
    cc = ca / cb;                     cc2 = MAGMA_C_DIV( ca2, cb2 );     check( fabs( cc - cc2 ) < stol );

    sa = fabs( ca );                  sa2 = MAGMA_C_ABS(  ca2 );         check( sa == sa2 );
    sa = abs1( ca );                  sa2 = MAGMA_C_ABS1( ca2 );         check( sa == sa2 );
    sa = real( ca );                  sa2 = MAGMA_C_REAL( ca2 );         check( sa == sa2 );
    sa = imag( ca );                  sa2 = MAGMA_C_IMAG( ca2 );         check( sa == sa2 );

    cc = ca;                          cc2 = ca2;                         check( cc == cc2 );
    eq = (ca == cb);                  eq2 = MAGMA_C_EQUAL( ca2, cb2 );   check( eq == eq2 );
    eq = (ca == cc);                  eq2 = MAGMA_C_EQUAL( ca2, cc2 );   check( eq == eq2 );
    printf( "magmaFloatComplex  operators    %s\n", (s == gStatus ? "ok" : "failed"));

    // --------------------
    // MAGMA operator                 MAGMA MACRO                        verify
    s = gStatus;
    da = 1.23;                        da2 = MAGMA_D_MAKE( 1.23, 2.45 );  check( da == da2 );
    db = 3.14;                        db2 = MAGMA_D_MAKE( 3.14, 2.72 );  check( db == db2 );
    dc = conj( da );                  dc2 = MAGMA_D_CONJ( da2 );         check( dc == dc2 );
    dc = -da;                         dc2 = MAGMA_D_NEGATE( da2 );       check( dc == dc2 );
    dc = da + db;                     dc2 = MAGMA_D_ADD( da2, db2 );     check( dc == dc2 );
    dc = da - db;                     dc2 = MAGMA_D_SUB( da2, db2 );     check( dc == dc2 );
    dc = da * db;                     dc2 = MAGMA_D_MUL( da2, db2 );     check( dc == dc2 );
    dc = da / db;                     dc2 = MAGMA_D_DIV( da2, db2 );     check( dc == dc2 );

    da = fabs( da );                  da2 = MAGMA_D_ABS(  da2 );         check( da == da2 );
    da = abs1( da );                  da2 = MAGMA_D_ABS1( da2 );         check( da == da2 );
    da = real( da );                  da2 = MAGMA_D_REAL( da2 );         check( da == da2 );
    da = imag( da );                  da2 = MAGMA_D_IMAG( da2 );         check( da == da2 );

    dc = da;                          dc2 = da2;                         check( dc == dc2 );
    eq = (da == db);                  eq2 = MAGMA_D_EQUAL( da2, db2 );   check( eq == eq2 );
    eq = (da == dc);                  eq2 = MAGMA_D_EQUAL( da2, dc2 );   check( eq == eq2 );
    printf( "double operators                %s\n", (s == gStatus ? "ok" : "failed"));

    // --------------------
    // MAGMA operator                 MAGMA MACRO                        verify
    s = gStatus;
    sa = MAGMA_S_MAKE( 1.23, 2.45 );  sa2 = MAGMA_S_MAKE( 1.23, 2.45 );  check( sa == sa2 );
    sb = MAGMA_S_MAKE( 3.14, 2.72 );  sb2 = MAGMA_S_MAKE( 3.14, 2.72 );  check( sb == sb2 );
    sc = conj( sa );                  sc2 = MAGMA_S_CONJ( sa2 );         check( sc == sc2 );
    sc = -sa;                         sc2 = MAGMA_S_NEGATE( sa2 );       check( sc == sc2 );
    sc = sa + sb;                     sc2 = MAGMA_S_ADD( sa2, sb2 );     check( sc == sc2 );
    sc = sa - sb;                     sc2 = MAGMA_S_SUB( sa2, sb2 );     check( sc == sc2 );
    sc = sa * sb;                     sc2 = MAGMA_S_MUL( sa2, sb2 );     check( sc == sc2 );
    sc = sa / sb;                     sc2 = MAGMA_S_DIV( sa2, sb2 );     check( sc == sc2 );

    sa = fabs( sa );                  sa2 = MAGMA_S_ABS(  sa2 );         check( sa == sa2 );
    sa = abs1( sa );                  sa2 = MAGMA_S_ABS1( sa2 );         check( sa == sa2 );
    sa = real( sa );                  sa2 = MAGMA_S_REAL( sa2 );         check( sa == sa2 );
    sa = imag( sa );                  sa2 = MAGMA_S_IMAG( sa2 );         check( sa == sa2 );

    sc = sa;                          sc2 = sa2;                         check( sc == sc2 );
    eq = (sa == sb);                  eq2 = MAGMA_S_EQUAL( sa2, sb2 );   check( eq == eq2 );
    eq = (sa == sc);                  eq2 = MAGMA_S_EQUAL( sa2, sc2 );   check( eq == eq2 );
    printf( "float  operators                %s\n", (s == gStatus ? "ok" : "failed"));

    // --------------------
    test_sqrt( dtol, stol );
    
    opts.cleanup();
    TESTING_FINALIZE();
    return gStatus;
}
