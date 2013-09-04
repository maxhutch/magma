
///////////////////////////////////////////////////////////////////////////////////////////////////

#define COMPLEX
#define DOUBLE
#define TEXTURE_1D

///////////////////////////////////////////////////////////////////////////////////////////////////

#ifdef COMPLEX
  #ifdef DOUBLE
    typedef magmaDoubleComplex FloatingPoint_t;
  #else
    typedef magmaFloatComplex FloatingPoint_t;
  #endif
#else
  #ifdef DOUBLE
    typedef double FloatingPoint_t;
  #else
    typedef float FloatingPoint_t;
  #endif
#endif

///////////////////////////////////////////////////////////////////////////////////////////////////

  #ifdef TEXTURE_1D

    #ifdef COMPLEX
      #ifdef DOUBLE
        static __device__
        FloatingPoint_t tex_fetch(texture<int4> tex_ref, int coord)
        {
            int4 v = tex1Dfetch(tex_ref, coord);
            return make_cuDoubleComplex(__hiloint2double(v.y, v.x), __hiloint2double(v.w, v.z));
        }
      #else
        static __device__
        FloatingPoint_t tex_fetch(texture<float2> tex_ref, int coord)
        {
            return tex1Dfetch(tex_ref, coord);
        }
      #endif
    #else
      #ifdef DOUBLE
        static __device__
        FloatingPoint_t tex_fetch(texture<int2> tex_ref, int coord)
        {
            int2 v = tex1Dfetch(tex_ref, coord);
            return __hiloint2double(v.y, v.x);
        }
      #else
        static __device__
        FloatingPoint_t tex_fetch(texture<float> tex_ref, int coord)
        {
            return tex1Dfetch(tex_ref, coord);
        }
      #endif
    #endif

  #endif

///////////////////////////////////////////////////////////////////////////////////////////////////
  #ifdef TEXTURE_1D
    #define fetch(A, m, n) tex_fetch(tex_ref_##A, coord_##A + n*LD##A+m)
  #else
    #define fetch(A, m, n) offs_d##A[n*LD##A+m]
  #endif

#ifdef COMPLEX
  #ifdef DOUBLE
    #define conj(A)          cuConj(A)
    #define add(A, B)        cuCadd(A, B)
    #define mul(A, B)        cuCmul(A, B)
    #define fma(A, B, C) C = cuCfma(A, B, C)
    #define make_FloatingPoint(x, y) make_cuDoubleComplex(x, y);
  #else
    #define conj(A)          cuConjf(A)
    #define add(A, B)        cuCaddf(A, B)
    #define mul(A, B)        cuCmulf(A, B)
    #define fma(A, B, C) C = cuCfmaf(A, B, C)
    #define make_FloatingPoint(x, y) make_cuFloatComplex(x, y);
  #endif
#else
    #define conj(A)           (A)
    #define add(A, B)         (A+B)
    #define mul(A, B)         (A*B)
    #define fma(A, B, C) C += (A*B)
    #define make_FloatingPoint(x, y) (x)
#endif

///////////////////////////////////////////////////////////////////////////////////////////////////
