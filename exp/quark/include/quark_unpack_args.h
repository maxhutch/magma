/**
 *
 * @file quark_unpack_args.h
 *
 * Macros used to retrieve arguments from QUARK to the function being executed.
 *
 * PLASMA is a software package provided by Univ. of Tennessee,
 * Univ. of California Berkeley and Univ. of Colorado Denver
 *
 * @version 2.3.1
 * @author Asim YarKhan
 * @date January 2015
 *
 **/

///////////////////////////////////////////////////////////////////////////////////////////////////
#ifndef _UNPACK_ARGS_
#define _UNPACK_ARGS_

#include "quark.h"

#define quark_unpack_args_1(quark, \
    arg1) \
{ \
  void *lastarg = NULL; \
  void *args_list = (void *)QUARK_Args_List( quark );	\
  void *arg = QUARK_Args_Pop(args_list, &lastarg); \
  memcpy(&arg1, arg, sizeof(arg1)); \
}

#define quark_unpack_args_2(quark, \
    arg1, \
    arg2) \
{ \
  void *lastarg = NULL; \
  void *args_list = QUARK_Args_List( quark ); \
  void *arg = QUARK_Args_Pop(args_list, &lastarg); \
  memcpy(&arg1, arg, sizeof(arg1)); arg = QUARK_Args_Pop(args_list, &lastarg); \
  memcpy(&arg2, arg, sizeof(arg2)); \
}

#define quark_unpack_args_3(quark, \
    arg1, \
    arg2, \
    arg3) \
{ \
  void *lastarg = NULL; \
  void *args_list = QUARK_Args_List( quark ); \
  void *arg = QUARK_Args_Pop(args_list, &lastarg); \
  memcpy(&arg1, arg, sizeof(arg1)); arg = QUARK_Args_Pop(args_list, &lastarg); \
  memcpy(&arg2, arg, sizeof(arg2)); arg = QUARK_Args_Pop(args_list, &lastarg); \
  memcpy(&arg3, arg, sizeof(arg3)); \
}

#define quark_unpack_args_4(quark, \
    arg1, \
    arg2, \
    arg3, \
    arg4) \
{ \
  void *lastarg = NULL; \
  void *args_list = QUARK_Args_List( quark ); \
  void *arg = QUARK_Args_Pop(args_list, &lastarg); \
  memcpy(&arg1, arg, sizeof(arg1)); arg = QUARK_Args_Pop(args_list, &lastarg); \
  memcpy(&arg2, arg, sizeof(arg2)); arg = QUARK_Args_Pop(args_list, &lastarg); \
  memcpy(&arg3, arg, sizeof(arg3)); arg = QUARK_Args_Pop(args_list, &lastarg); \
  memcpy(&arg4, arg, sizeof(arg4)); \
}

#define quark_unpack_args_5(quark, \
    arg1, \
    arg2, \
    arg3, \
    arg4, \
    arg5) \
{ \
  void *lastarg = NULL; \
  void *args_list = QUARK_Args_List( quark ); \
  void *arg = QUARK_Args_Pop(args_list, &lastarg); \
  memcpy(&arg1, arg, sizeof(arg1)); arg = QUARK_Args_Pop(args_list, &lastarg); \
  memcpy(&arg2, arg, sizeof(arg2)); arg = QUARK_Args_Pop(args_list, &lastarg); \
  memcpy(&arg3, arg, sizeof(arg3)); arg = QUARK_Args_Pop(args_list, &lastarg); \
  memcpy(&arg4, arg, sizeof(arg4)); arg = QUARK_Args_Pop(args_list, &lastarg); \
  memcpy(&arg5, arg, sizeof(arg5)); \
}

#define quark_unpack_args_6(quark, \
    arg1, \
    arg2, \
    arg3, \
    arg4, \
    arg5, \
    arg6) \
{ \
  void *lastarg = NULL; \
  void *args_list = QUARK_Args_List( quark ); \
  void *arg = QUARK_Args_Pop(args_list, &lastarg); \
  memcpy(&arg1, arg, sizeof(arg1)); arg = QUARK_Args_Pop(args_list, &lastarg); \
  memcpy(&arg2, arg, sizeof(arg2)); arg = QUARK_Args_Pop(args_list, &lastarg); \
  memcpy(&arg3, arg, sizeof(arg3)); arg = QUARK_Args_Pop(args_list, &lastarg); \
  memcpy(&arg4, arg, sizeof(arg4)); arg = QUARK_Args_Pop(args_list, &lastarg); \
  memcpy(&arg5, arg, sizeof(arg5)); arg = QUARK_Args_Pop(args_list, &lastarg); \
  memcpy(&arg6, arg, sizeof(arg6)); \
}

#define quark_unpack_args_7(quark, \
    arg1, \
    arg2, \
    arg3, \
    arg4, \
    arg5, \
    arg6, \
    arg7) \
{ \
  void *lastarg = NULL; \
  void *args_list = QUARK_Args_List( quark ); \
  void *arg = QUARK_Args_Pop(args_list, &lastarg); \
  memcpy(&arg1, arg, sizeof(arg1)); arg = QUARK_Args_Pop(args_list, &lastarg); \
  memcpy(&arg2, arg, sizeof(arg2)); arg = QUARK_Args_Pop(args_list, &lastarg); \
  memcpy(&arg3, arg, sizeof(arg3)); arg = QUARK_Args_Pop(args_list, &lastarg); \
  memcpy(&arg4, arg, sizeof(arg4)); arg = QUARK_Args_Pop(args_list, &lastarg); \
  memcpy(&arg5, arg, sizeof(arg5)); arg = QUARK_Args_Pop(args_list, &lastarg); \
  memcpy(&arg6, arg, sizeof(arg6)); arg = QUARK_Args_Pop(args_list, &lastarg); \
  memcpy(&arg7, arg, sizeof(arg7)); \
}

#define quark_unpack_args_8(quark, \
    arg1, \
    arg2, \
    arg3, \
    arg4, \
    arg5, \
    arg6, \
    arg7, \
    arg8) \
{ \
  void *lastarg = NULL; \
  void *args_list = QUARK_Args_List( quark ); \
  void *arg = QUARK_Args_Pop(args_list, &lastarg); \
  memcpy(&arg1, arg, sizeof(arg1)); arg = QUARK_Args_Pop(args_list, &lastarg); \
  memcpy(&arg2, arg, sizeof(arg2)); arg = QUARK_Args_Pop(args_list, &lastarg); \
  memcpy(&arg3, arg, sizeof(arg3)); arg = QUARK_Args_Pop(args_list, &lastarg); \
  memcpy(&arg4, arg, sizeof(arg4)); arg = QUARK_Args_Pop(args_list, &lastarg); \
  memcpy(&arg5, arg, sizeof(arg5)); arg = QUARK_Args_Pop(args_list, &lastarg); \
  memcpy(&arg6, arg, sizeof(arg6)); arg = QUARK_Args_Pop(args_list, &lastarg); \
  memcpy(&arg7, arg, sizeof(arg7)); arg = QUARK_Args_Pop(args_list, &lastarg); \
  memcpy(&arg8, arg, sizeof(arg8)); \
}

#define quark_unpack_args_9(quark, \
    arg1, \
    arg2, \
    arg3, \
    arg4, \
    arg5, \
    arg6, \
    arg7, \
    arg8, \
    arg9) \
{ \
  void *lastarg = NULL; \
  void *args_list = QUARK_Args_List( quark ); \
  void *arg = QUARK_Args_Pop(args_list, &lastarg); \
  memcpy(&arg1, arg, sizeof(arg1)); arg = QUARK_Args_Pop(args_list, &lastarg); \
  memcpy(&arg2, arg, sizeof(arg2)); arg = QUARK_Args_Pop(args_list, &lastarg); \
  memcpy(&arg3, arg, sizeof(arg3)); arg = QUARK_Args_Pop(args_list, &lastarg); \
  memcpy(&arg4, arg, sizeof(arg4)); arg = QUARK_Args_Pop(args_list, &lastarg); \
  memcpy(&arg5, arg, sizeof(arg5)); arg = QUARK_Args_Pop(args_list, &lastarg); \
  memcpy(&arg6, arg, sizeof(arg6)); arg = QUARK_Args_Pop(args_list, &lastarg); \
  memcpy(&arg7, arg, sizeof(arg7)); arg = QUARK_Args_Pop(args_list, &lastarg); \
  memcpy(&arg8, arg, sizeof(arg8)); arg = QUARK_Args_Pop(args_list, &lastarg); \
  memcpy(&arg9, arg, sizeof(arg9)); \
}

#define quark_unpack_args_10(quark, \
    arg1, \
    arg2, \
    arg3, \
    arg4, \
    arg5, \
    arg6, \
    arg7, \
    arg8, \
    arg9, \
    arg10) \
{ \
  void *lastarg = NULL; \
  void *args_list = QUARK_Args_List( quark ); \
  void *arg = QUARK_Args_Pop(args_list, &lastarg); \
  memcpy(&arg1, arg, sizeof(arg1)); arg = QUARK_Args_Pop(args_list, &lastarg); \
  memcpy(&arg2, arg, sizeof(arg2)); arg = QUARK_Args_Pop(args_list, &lastarg); \
  memcpy(&arg3, arg, sizeof(arg3)); arg = QUARK_Args_Pop(args_list, &lastarg); \
  memcpy(&arg4, arg, sizeof(arg4)); arg = QUARK_Args_Pop(args_list, &lastarg); \
  memcpy(&arg5, arg, sizeof(arg5)); arg = QUARK_Args_Pop(args_list, &lastarg); \
  memcpy(&arg6, arg, sizeof(arg6)); arg = QUARK_Args_Pop(args_list, &lastarg); \
  memcpy(&arg7, arg, sizeof(arg7)); arg = QUARK_Args_Pop(args_list, &lastarg); \
  memcpy(&arg8, arg, sizeof(arg8)); arg = QUARK_Args_Pop(args_list, &lastarg); \
  memcpy(&arg9, arg, sizeof(arg9)); arg = QUARK_Args_Pop(args_list, &lastarg); \
  memcpy(&arg10, arg, sizeof(arg10)); \
}

#define quark_unpack_args_11(quark, \
    arg1, \
    arg2, \
    arg3, \
    arg4, \
    arg5, \
    arg6, \
    arg7, \
    arg8, \
    arg9, \
    arg10, \
    arg11) \
{ \
  void *lastarg = NULL; \
  void *args_list = QUARK_Args_List( quark ); \
  void *arg = QUARK_Args_Pop(args_list, &lastarg); \
  memcpy(&arg1, arg, sizeof(arg1)); arg = QUARK_Args_Pop(args_list, &lastarg); \
  memcpy(&arg2, arg, sizeof(arg2)); arg = QUARK_Args_Pop(args_list, &lastarg); \
  memcpy(&arg3, arg, sizeof(arg3)); arg = QUARK_Args_Pop(args_list, &lastarg); \
  memcpy(&arg4, arg, sizeof(arg4)); arg = QUARK_Args_Pop(args_list, &lastarg); \
  memcpy(&arg5, arg, sizeof(arg5)); arg = QUARK_Args_Pop(args_list, &lastarg); \
  memcpy(&arg6, arg, sizeof(arg6)); arg = QUARK_Args_Pop(args_list, &lastarg); \
  memcpy(&arg7, arg, sizeof(arg7)); arg = QUARK_Args_Pop(args_list, &lastarg); \
  memcpy(&arg8, arg, sizeof(arg8)); arg = QUARK_Args_Pop(args_list, &lastarg); \
  memcpy(&arg9, arg, sizeof(arg9)); arg = QUARK_Args_Pop(args_list, &lastarg); \
  memcpy(&arg10, arg, sizeof(arg10)); arg = QUARK_Args_Pop(args_list, &lastarg); \
  memcpy(&arg11, arg, sizeof(arg11)); \
}

#define quark_unpack_args_12(quark, \
    arg1, \
    arg2, \
    arg3, \
    arg4, \
    arg5, \
    arg6, \
    arg7, \
    arg8, \
    arg9, \
    arg10, \
    arg11, \
    arg12) \
{ \
  void *lastarg = NULL; \
  void *args_list = QUARK_Args_List( quark ); \
  void *arg = QUARK_Args_Pop(args_list, &lastarg); \
  memcpy(&arg1, arg, sizeof(arg1)); arg = QUARK_Args_Pop(args_list, &lastarg); \
  memcpy(&arg2, arg, sizeof(arg2)); arg = QUARK_Args_Pop(args_list, &lastarg); \
  memcpy(&arg3, arg, sizeof(arg3)); arg = QUARK_Args_Pop(args_list, &lastarg); \
  memcpy(&arg4, arg, sizeof(arg4)); arg = QUARK_Args_Pop(args_list, &lastarg); \
  memcpy(&arg5, arg, sizeof(arg5)); arg = QUARK_Args_Pop(args_list, &lastarg); \
  memcpy(&arg6, arg, sizeof(arg6)); arg = QUARK_Args_Pop(args_list, &lastarg); \
  memcpy(&arg7, arg, sizeof(arg7)); arg = QUARK_Args_Pop(args_list, &lastarg); \
  memcpy(&arg8, arg, sizeof(arg8)); arg = QUARK_Args_Pop(args_list, &lastarg); \
  memcpy(&arg9, arg, sizeof(arg9)); arg = QUARK_Args_Pop(args_list, &lastarg); \
  memcpy(&arg10, arg, sizeof(arg10)); arg = QUARK_Args_Pop(args_list, &lastarg); \
  memcpy(&arg11, arg, sizeof(arg11)); arg = QUARK_Args_Pop(args_list, &lastarg); \
  memcpy(&arg12, arg, sizeof(arg12)); \
}

#define quark_unpack_args_13(quark, \
    arg1, \
    arg2, \
    arg3, \
    arg4, \
    arg5, \
    arg6, \
    arg7, \
    arg8, \
    arg9, \
    arg10, \
    arg11, \
    arg12, \
    arg13) \
{ \
  void *lastarg = NULL; \
  void *args_list = QUARK_Args_List( quark ); \
  void *arg = QUARK_Args_Pop(args_list, &lastarg); \
  memcpy(&arg1, arg, sizeof(arg1)); arg = QUARK_Args_Pop(args_list, &lastarg); \
  memcpy(&arg2, arg, sizeof(arg2)); arg = QUARK_Args_Pop(args_list, &lastarg); \
  memcpy(&arg3, arg, sizeof(arg3)); arg = QUARK_Args_Pop(args_list, &lastarg); \
  memcpy(&arg4, arg, sizeof(arg4)); arg = QUARK_Args_Pop(args_list, &lastarg); \
  memcpy(&arg5, arg, sizeof(arg5)); arg = QUARK_Args_Pop(args_list, &lastarg); \
  memcpy(&arg6, arg, sizeof(arg6)); arg = QUARK_Args_Pop(args_list, &lastarg); \
  memcpy(&arg7, arg, sizeof(arg7)); arg = QUARK_Args_Pop(args_list, &lastarg); \
  memcpy(&arg8, arg, sizeof(arg8)); arg = QUARK_Args_Pop(args_list, &lastarg); \
  memcpy(&arg9, arg, sizeof(arg9)); arg = QUARK_Args_Pop(args_list, &lastarg); \
  memcpy(&arg10, arg, sizeof(arg10)); arg = QUARK_Args_Pop(args_list, &lastarg); \
  memcpy(&arg11, arg, sizeof(arg11)); arg = QUARK_Args_Pop(args_list, &lastarg); \
  memcpy(&arg12, arg, sizeof(arg12)); arg = QUARK_Args_Pop(args_list, &lastarg); \
  memcpy(&arg13, arg, sizeof(arg13)); \
}

#define quark_unpack_args_14(quark, \
    arg1, \
    arg2, \
    arg3, \
    arg4, \
    arg5, \
    arg6, \
    arg7, \
    arg8, \
    arg9, \
    arg10, \
    arg11, \
    arg12, \
    arg13, \
    arg14) \
{ \
  void *lastarg = NULL; \
  void *args_list = QUARK_Args_List( quark ); \
  void *arg = QUARK_Args_Pop(args_list, &lastarg); \
  memcpy(&arg1, arg, sizeof(arg1)); arg = QUARK_Args_Pop(args_list, &lastarg); \
  memcpy(&arg2, arg, sizeof(arg2)); arg = QUARK_Args_Pop(args_list, &lastarg); \
  memcpy(&arg3, arg, sizeof(arg3)); arg = QUARK_Args_Pop(args_list, &lastarg); \
  memcpy(&arg4, arg, sizeof(arg4)); arg = QUARK_Args_Pop(args_list, &lastarg); \
  memcpy(&arg5, arg, sizeof(arg5)); arg = QUARK_Args_Pop(args_list, &lastarg); \
  memcpy(&arg6, arg, sizeof(arg6)); arg = QUARK_Args_Pop(args_list, &lastarg); \
  memcpy(&arg7, arg, sizeof(arg7)); arg = QUARK_Args_Pop(args_list, &lastarg); \
  memcpy(&arg8, arg, sizeof(arg8)); arg = QUARK_Args_Pop(args_list, &lastarg); \
  memcpy(&arg9, arg, sizeof(arg9)); arg = QUARK_Args_Pop(args_list, &lastarg); \
  memcpy(&arg10, arg, sizeof(arg10)); arg = QUARK_Args_Pop(args_list, &lastarg); \
  memcpy(&arg11, arg, sizeof(arg11)); arg = QUARK_Args_Pop(args_list, &lastarg); \
  memcpy(&arg12, arg, sizeof(arg12)); arg = QUARK_Args_Pop(args_list, &lastarg); \
  memcpy(&arg13, arg, sizeof(arg13)); arg = QUARK_Args_Pop(args_list, &lastarg); \
  memcpy(&arg14, arg, sizeof(arg14)); \
}

#define quark_unpack_args_15(quark, \
    arg1, \
    arg2, \
    arg3, \
    arg4, \
    arg5, \
    arg6, \
    arg7, \
    arg8, \
    arg9, \
    arg10, \
    arg11, \
    arg12, \
    arg13, \
    arg14, \
    arg15) \
{ \
  void *lastarg = NULL; \
  void *args_list = QUARK_Args_List( quark ); \
  void *arg = QUARK_Args_Pop(args_list, &lastarg); \
  memcpy(&arg1, arg, sizeof(arg1)); arg = QUARK_Args_Pop(args_list, &lastarg); \
  memcpy(&arg2, arg, sizeof(arg2)); arg = QUARK_Args_Pop(args_list, &lastarg); \
  memcpy(&arg3, arg, sizeof(arg3)); arg = QUARK_Args_Pop(args_list, &lastarg); \
  memcpy(&arg4, arg, sizeof(arg4)); arg = QUARK_Args_Pop(args_list, &lastarg); \
  memcpy(&arg5, arg, sizeof(arg5)); arg = QUARK_Args_Pop(args_list, &lastarg); \
  memcpy(&arg6, arg, sizeof(arg6)); arg = QUARK_Args_Pop(args_list, &lastarg); \
  memcpy(&arg7, arg, sizeof(arg7)); arg = QUARK_Args_Pop(args_list, &lastarg); \
  memcpy(&arg8, arg, sizeof(arg8)); arg = QUARK_Args_Pop(args_list, &lastarg); \
  memcpy(&arg9, arg, sizeof(arg9)); arg = QUARK_Args_Pop(args_list, &lastarg); \
  memcpy(&arg10, arg, sizeof(arg10)); arg = QUARK_Args_Pop(args_list, &lastarg); \
  memcpy(&arg11, arg, sizeof(arg11)); arg = QUARK_Args_Pop(args_list, &lastarg); \
  memcpy(&arg12, arg, sizeof(arg12)); arg = QUARK_Args_Pop(args_list, &lastarg); \
  memcpy(&arg13, arg, sizeof(arg13)); arg = QUARK_Args_Pop(args_list, &lastarg); \
  memcpy(&arg14, arg, sizeof(arg14)); arg = QUARK_Args_Pop(args_list, &lastarg); \
  memcpy(&arg15, arg, sizeof(arg15)); \
}

#define quark_unpack_args_16(quark, \
    arg1, \
    arg2, \
    arg3, \
    arg4, \
    arg5, \
    arg6, \
    arg7, \
    arg8, \
    arg9, \
    arg10, \
    arg11, \
    arg12, \
    arg13, \
    arg14, \
    arg15, \
    arg16) \
{ \
  void *lastarg = NULL; \
  void *args_list = QUARK_Args_List( quark ); \
  void *arg = QUARK_Args_Pop(args_list, &lastarg); \
  memcpy(&arg1, arg, sizeof(arg1)); arg = QUARK_Args_Pop(args_list, &lastarg); \
  memcpy(&arg2, arg, sizeof(arg2)); arg = QUARK_Args_Pop(args_list, &lastarg); \
  memcpy(&arg3, arg, sizeof(arg3)); arg = QUARK_Args_Pop(args_list, &lastarg); \
  memcpy(&arg4, arg, sizeof(arg4)); arg = QUARK_Args_Pop(args_list, &lastarg); \
  memcpy(&arg5, arg, sizeof(arg5)); arg = QUARK_Args_Pop(args_list, &lastarg); \
  memcpy(&arg6, arg, sizeof(arg6)); arg = QUARK_Args_Pop(args_list, &lastarg); \
  memcpy(&arg7, arg, sizeof(arg7)); arg = QUARK_Args_Pop(args_list, &lastarg); \
  memcpy(&arg8, arg, sizeof(arg8)); arg = QUARK_Args_Pop(args_list, &lastarg); \
  memcpy(&arg9, arg, sizeof(arg9)); arg = QUARK_Args_Pop(args_list, &lastarg); \
  memcpy(&arg10, arg, sizeof(arg10)); arg = QUARK_Args_Pop(args_list, &lastarg); \
  memcpy(&arg11, arg, sizeof(arg11)); arg = QUARK_Args_Pop(args_list, &lastarg); \
  memcpy(&arg12, arg, sizeof(arg12)); arg = QUARK_Args_Pop(args_list, &lastarg); \
  memcpy(&arg13, arg, sizeof(arg13)); arg = QUARK_Args_Pop(args_list, &lastarg); \
  memcpy(&arg14, arg, sizeof(arg14)); arg = QUARK_Args_Pop(args_list, &lastarg); \
  memcpy(&arg15, arg, sizeof(arg15)); arg = QUARK_Args_Pop(args_list, &lastarg); \
  memcpy(&arg16, arg, sizeof(arg16)); \
}

#define quark_unpack_args_17(quark, \
    arg1, \
    arg2, \
    arg3, \
    arg4, \
    arg5, \
    arg6, \
    arg7, \
    arg8, \
    arg9, \
    arg10, \
    arg11, \
    arg12, \
    arg13, \
    arg14, \
    arg15, \
    arg16, \
    arg17) \
{ \
  void *lastarg = NULL; \
  void *args_list = QUARK_Args_List( quark ); \
  void *arg = QUARK_Args_Pop(args_list, &lastarg); \
  memcpy(&arg1, arg, sizeof(arg1)); arg = QUARK_Args_Pop(args_list, &lastarg); \
  memcpy(&arg2, arg, sizeof(arg2)); arg = QUARK_Args_Pop(args_list, &lastarg); \
  memcpy(&arg3, arg, sizeof(arg3)); arg = QUARK_Args_Pop(args_list, &lastarg); \
  memcpy(&arg4, arg, sizeof(arg4)); arg = QUARK_Args_Pop(args_list, &lastarg); \
  memcpy(&arg5, arg, sizeof(arg5)); arg = QUARK_Args_Pop(args_list, &lastarg); \
  memcpy(&arg6, arg, sizeof(arg6)); arg = QUARK_Args_Pop(args_list, &lastarg); \
  memcpy(&arg7, arg, sizeof(arg7)); arg = QUARK_Args_Pop(args_list, &lastarg); \
  memcpy(&arg8, arg, sizeof(arg8)); arg = QUARK_Args_Pop(args_list, &lastarg); \
  memcpy(&arg9, arg, sizeof(arg9)); arg = QUARK_Args_Pop(args_list, &lastarg); \
  memcpy(&arg10, arg, sizeof(arg10)); arg = QUARK_Args_Pop(args_list, &lastarg); \
  memcpy(&arg11, arg, sizeof(arg11)); arg = QUARK_Args_Pop(args_list, &lastarg); \
  memcpy(&arg12, arg, sizeof(arg12)); arg = QUARK_Args_Pop(args_list, &lastarg); \
  memcpy(&arg13, arg, sizeof(arg13)); arg = QUARK_Args_Pop(args_list, &lastarg); \
  memcpy(&arg14, arg, sizeof(arg14)); arg = QUARK_Args_Pop(args_list, &lastarg); \
  memcpy(&arg15, arg, sizeof(arg15)); arg = QUARK_Args_Pop(args_list, &lastarg); \
  memcpy(&arg16, arg, sizeof(arg16)); arg = QUARK_Args_Pop(args_list, &lastarg); \
  memcpy(&arg17, arg, sizeof(arg17)); \
}

#define quark_unpack_args_18(quark, \
    arg1, \
    arg2, \
    arg3, \
    arg4, \
    arg5, \
    arg6, \
    arg7, \
    arg8, \
    arg9, \
    arg10, \
    arg11, \
    arg12, \
    arg13, \
    arg14, \
    arg15, \
    arg16, \
    arg17, \
    arg18) \
{ \
  void *lastarg = NULL; \
  void *args_list = QUARK_Args_List( quark ); \
  void *arg = QUARK_Args_Pop(args_list, &lastarg); \
  memcpy(&arg1, arg, sizeof(arg1)); arg = QUARK_Args_Pop(args_list, &lastarg); \
  memcpy(&arg2, arg, sizeof(arg2)); arg = QUARK_Args_Pop(args_list, &lastarg); \
  memcpy(&arg3, arg, sizeof(arg3)); arg = QUARK_Args_Pop(args_list, &lastarg); \
  memcpy(&arg4, arg, sizeof(arg4)); arg = QUARK_Args_Pop(args_list, &lastarg); \
  memcpy(&arg5, arg, sizeof(arg5)); arg = QUARK_Args_Pop(args_list, &lastarg); \
  memcpy(&arg6, arg, sizeof(arg6)); arg = QUARK_Args_Pop(args_list, &lastarg); \
  memcpy(&arg7, arg, sizeof(arg7)); arg = QUARK_Args_Pop(args_list, &lastarg); \
  memcpy(&arg8, arg, sizeof(arg8)); arg = QUARK_Args_Pop(args_list, &lastarg); \
  memcpy(&arg9, arg, sizeof(arg9)); arg = QUARK_Args_Pop(args_list, &lastarg); \
  memcpy(&arg10, arg, sizeof(arg10)); arg = QUARK_Args_Pop(args_list, &lastarg); \
  memcpy(&arg11, arg, sizeof(arg11)); arg = QUARK_Args_Pop(args_list, &lastarg); \
  memcpy(&arg12, arg, sizeof(arg12)); arg = QUARK_Args_Pop(args_list, &lastarg); \
  memcpy(&arg13, arg, sizeof(arg13)); arg = QUARK_Args_Pop(args_list, &lastarg); \
  memcpy(&arg14, arg, sizeof(arg14)); arg = QUARK_Args_Pop(args_list, &lastarg); \
  memcpy(&arg15, arg, sizeof(arg15)); arg = QUARK_Args_Pop(args_list, &lastarg); \
  memcpy(&arg16, arg, sizeof(arg16)); arg = QUARK_Args_Pop(args_list, &lastarg); \
  memcpy(&arg17, arg, sizeof(arg17)); arg = QUARK_Args_Pop(args_list, &lastarg); \
  memcpy(&arg18, arg, sizeof(arg18)); \
}

#define quark_unpack_args_19(quark, \
    arg1, \
    arg2, \
    arg3, \
    arg4, \
    arg5, \
    arg6, \
    arg7, \
    arg8, \
    arg9, \
    arg10, \
    arg11, \
    arg12, \
    arg13, \
    arg14, \
    arg15, \
    arg16, \
    arg17, \
    arg18, \
    arg19) \
{ \
  void *lastarg = NULL; \
  void *args_list = QUARK_Args_List( quark ); \
  void *arg = QUARK_Args_Pop(args_list, &lastarg); \
  memcpy(&arg1, arg, sizeof(arg1)); arg = QUARK_Args_Pop(args_list, &lastarg); \
  memcpy(&arg2, arg, sizeof(arg2)); arg = QUARK_Args_Pop(args_list, &lastarg); \
  memcpy(&arg3, arg, sizeof(arg3)); arg = QUARK_Args_Pop(args_list, &lastarg); \
  memcpy(&arg4, arg, sizeof(arg4)); arg = QUARK_Args_Pop(args_list, &lastarg); \
  memcpy(&arg5, arg, sizeof(arg5)); arg = QUARK_Args_Pop(args_list, &lastarg); \
  memcpy(&arg6, arg, sizeof(arg6)); arg = QUARK_Args_Pop(args_list, &lastarg); \
  memcpy(&arg7, arg, sizeof(arg7)); arg = QUARK_Args_Pop(args_list, &lastarg); \
  memcpy(&arg8, arg, sizeof(arg8)); arg = QUARK_Args_Pop(args_list, &lastarg); \
  memcpy(&arg9, arg, sizeof(arg9)); arg = QUARK_Args_Pop(args_list, &lastarg); \
  memcpy(&arg10, arg, sizeof(arg10)); arg = QUARK_Args_Pop(args_list, &lastarg); \
  memcpy(&arg11, arg, sizeof(arg11)); arg = QUARK_Args_Pop(args_list, &lastarg); \
  memcpy(&arg12, arg, sizeof(arg12)); arg = QUARK_Args_Pop(args_list, &lastarg); \
  memcpy(&arg13, arg, sizeof(arg13)); arg = QUARK_Args_Pop(args_list, &lastarg); \
  memcpy(&arg14, arg, sizeof(arg14)); arg = QUARK_Args_Pop(args_list, &lastarg); \
  memcpy(&arg15, arg, sizeof(arg15)); arg = QUARK_Args_Pop(args_list, &lastarg); \
  memcpy(&arg16, arg, sizeof(arg16)); arg = QUARK_Args_Pop(args_list, &lastarg); \
  memcpy(&arg17, arg, sizeof(arg17)); arg = QUARK_Args_Pop(args_list, &lastarg); \
  memcpy(&arg18, arg, sizeof(arg18)); arg = QUARK_Args_Pop(args_list, &lastarg); \
  memcpy(&arg19, arg, sizeof(arg19)); \
}

#define quark_unpack_args_20(quark, \
    arg1, \
    arg2, \
    arg3, \
    arg4, \
    arg5, \
    arg6, \
    arg7, \
    arg8, \
    arg9, \
    arg10, \
    arg11, \
    arg12, \
    arg13, \
    arg14, \
    arg15, \
    arg16, \
    arg17, \
    arg18, \
    arg19, \
    arg20) \
{ \
  void *lastarg = NULL; \
  void *args_list = QUARK_Args_List( quark ); \
  void *arg = QUARK_Args_Pop(args_list, &lastarg); \
  memcpy(&arg1, arg, sizeof(arg1)); arg = QUARK_Args_Pop(args_list, &lastarg); \
  memcpy(&arg2, arg, sizeof(arg2)); arg = QUARK_Args_Pop(args_list, &lastarg); \
  memcpy(&arg3, arg, sizeof(arg3)); arg = QUARK_Args_Pop(args_list, &lastarg); \
  memcpy(&arg4, arg, sizeof(arg4)); arg = QUARK_Args_Pop(args_list, &lastarg); \
  memcpy(&arg5, arg, sizeof(arg5)); arg = QUARK_Args_Pop(args_list, &lastarg); \
  memcpy(&arg6, arg, sizeof(arg6)); arg = QUARK_Args_Pop(args_list, &lastarg); \
  memcpy(&arg7, arg, sizeof(arg7)); arg = QUARK_Args_Pop(args_list, &lastarg); \
  memcpy(&arg8, arg, sizeof(arg8)); arg = QUARK_Args_Pop(args_list, &lastarg); \
  memcpy(&arg9, arg, sizeof(arg9)); arg = QUARK_Args_Pop(args_list, &lastarg); \
  memcpy(&arg10, arg, sizeof(arg10)); arg = QUARK_Args_Pop(args_list, &lastarg); \
  memcpy(&arg11, arg, sizeof(arg11)); arg = QUARK_Args_Pop(args_list, &lastarg); \
  memcpy(&arg12, arg, sizeof(arg12)); arg = QUARK_Args_Pop(args_list, &lastarg); \
  memcpy(&arg13, arg, sizeof(arg13)); arg = QUARK_Args_Pop(args_list, &lastarg); \
  memcpy(&arg14, arg, sizeof(arg14)); arg = QUARK_Args_Pop(args_list, &lastarg); \
  memcpy(&arg15, arg, sizeof(arg15)); arg = QUARK_Args_Pop(args_list, &lastarg); \
  memcpy(&arg16, arg, sizeof(arg16)); arg = QUARK_Args_Pop(args_list, &lastarg); \
  memcpy(&arg17, arg, sizeof(arg17)); arg = QUARK_Args_Pop(args_list, &lastarg); \
  memcpy(&arg18, arg, sizeof(arg18)); arg = QUARK_Args_Pop(args_list, &lastarg); \
  memcpy(&arg19, arg, sizeof(arg19)); arg = QUARK_Args_Pop(args_list, &lastarg); \
  memcpy(&arg20, arg, sizeof(arg20)); \
}

#define quark_unpack_args_21(quark, \
    arg1, \
    arg2, \
    arg3, \
    arg4, \
    arg5, \
    arg6, \
    arg7, \
    arg8, \
    arg9, \
    arg10, \
    arg11, \
    arg12, \
    arg13, \
    arg14, \
    arg15, \
    arg16, \
    arg17, \
    arg18, \
    arg19, \
    arg20, \
    arg21) \
{ \
  void *lastarg = NULL; \
  void *args_list = QUARK_Args_List( quark ); \
  void *arg = QUARK_Args_Pop(args_list, &lastarg); \
  memcpy(&arg1, arg, sizeof(arg1)); arg = QUARK_Args_Pop(args_list, &lastarg); \
  memcpy(&arg2, arg, sizeof(arg2)); arg = QUARK_Args_Pop(args_list, &lastarg); \
  memcpy(&arg3, arg, sizeof(arg3)); arg = QUARK_Args_Pop(args_list, &lastarg); \
  memcpy(&arg4, arg, sizeof(arg4)); arg = QUARK_Args_Pop(args_list, &lastarg); \
  memcpy(&arg5, arg, sizeof(arg5)); arg = QUARK_Args_Pop(args_list, &lastarg); \
  memcpy(&arg6, arg, sizeof(arg6)); arg = QUARK_Args_Pop(args_list, &lastarg); \
  memcpy(&arg7, arg, sizeof(arg7)); arg = QUARK_Args_Pop(args_list, &lastarg); \
  memcpy(&arg8, arg, sizeof(arg8)); arg = QUARK_Args_Pop(args_list, &lastarg); \
  memcpy(&arg9, arg, sizeof(arg9)); arg = QUARK_Args_Pop(args_list, &lastarg); \
  memcpy(&arg10, arg, sizeof(arg10)); arg = QUARK_Args_Pop(args_list, &lastarg); \
  memcpy(&arg11, arg, sizeof(arg11)); arg = QUARK_Args_Pop(args_list, &lastarg); \
  memcpy(&arg12, arg, sizeof(arg12)); arg = QUARK_Args_Pop(args_list, &lastarg); \
  memcpy(&arg13, arg, sizeof(arg13)); arg = QUARK_Args_Pop(args_list, &lastarg); \
  memcpy(&arg14, arg, sizeof(arg14)); arg = QUARK_Args_Pop(args_list, &lastarg); \
  memcpy(&arg15, arg, sizeof(arg15)); arg = QUARK_Args_Pop(args_list, &lastarg); \
  memcpy(&arg16, arg, sizeof(arg16)); arg = QUARK_Args_Pop(args_list, &lastarg); \
  memcpy(&arg17, arg, sizeof(arg17)); arg = QUARK_Args_Pop(args_list, &lastarg); \
  memcpy(&arg18, arg, sizeof(arg18)); arg = QUARK_Args_Pop(args_list, &lastarg); \
  memcpy(&arg19, arg, sizeof(arg19)); arg = QUARK_Args_Pop(args_list, &lastarg); \
  memcpy(&arg20, arg, sizeof(arg20)); arg = QUARK_Args_Pop(args_list, &lastarg); \
  memcpy(&arg21, arg, sizeof(arg21)); \
}

#define quark_unpack_args_22(quark, \
    arg1, \
    arg2, \
    arg3, \
    arg4, \
    arg5, \
    arg6, \
    arg7, \
    arg8, \
    arg9, \
    arg10, \
    arg11, \
    arg12, \
    arg13, \
    arg14, \
    arg15, \
    arg16, \
    arg17, \
    arg18, \
    arg19, \
    arg20, \
    arg21, \
    arg22) \
{ \
  void *lastarg = NULL; \
  void *args_list = QUARK_Args_List( quark ); \
  void *arg = QUARK_Args_Pop(args_list, &lastarg); \
  memcpy(&arg1, arg, sizeof(arg1)); arg = QUARK_Args_Pop(args_list, &lastarg); \
  memcpy(&arg2, arg, sizeof(arg2)); arg = QUARK_Args_Pop(args_list, &lastarg); \
  memcpy(&arg3, arg, sizeof(arg3)); arg = QUARK_Args_Pop(args_list, &lastarg); \
  memcpy(&arg4, arg, sizeof(arg4)); arg = QUARK_Args_Pop(args_list, &lastarg); \
  memcpy(&arg5, arg, sizeof(arg5)); arg = QUARK_Args_Pop(args_list, &lastarg); \
  memcpy(&arg6, arg, sizeof(arg6)); arg = QUARK_Args_Pop(args_list, &lastarg); \
  memcpy(&arg7, arg, sizeof(arg7)); arg = QUARK_Args_Pop(args_list, &lastarg); \
  memcpy(&arg8, arg, sizeof(arg8)); arg = QUARK_Args_Pop(args_list, &lastarg); \
  memcpy(&arg9, arg, sizeof(arg9)); arg = QUARK_Args_Pop(args_list, &lastarg); \
  memcpy(&arg10, arg, sizeof(arg10)); arg = QUARK_Args_Pop(args_list, &lastarg); \
  memcpy(&arg11, arg, sizeof(arg11)); arg = QUARK_Args_Pop(args_list, &lastarg); \
  memcpy(&arg12, arg, sizeof(arg12)); arg = QUARK_Args_Pop(args_list, &lastarg); \
  memcpy(&arg13, arg, sizeof(arg13)); arg = QUARK_Args_Pop(args_list, &lastarg); \
  memcpy(&arg14, arg, sizeof(arg14)); arg = QUARK_Args_Pop(args_list, &lastarg); \
  memcpy(&arg15, arg, sizeof(arg15)); arg = QUARK_Args_Pop(args_list, &lastarg); \
  memcpy(&arg16, arg, sizeof(arg16)); arg = QUARK_Args_Pop(args_list, &lastarg); \
  memcpy(&arg17, arg, sizeof(arg17)); arg = QUARK_Args_Pop(args_list, &lastarg); \
  memcpy(&arg18, arg, sizeof(arg18)); arg = QUARK_Args_Pop(args_list, &lastarg); \
  memcpy(&arg19, arg, sizeof(arg19)); arg = QUARK_Args_Pop(args_list, &lastarg); \
  memcpy(&arg20, arg, sizeof(arg20)); arg = QUARK_Args_Pop(args_list, &lastarg); \
  memcpy(&arg21, arg, sizeof(arg21)); arg = QUARK_Args_Pop(args_list, &lastarg); \
  memcpy(&arg22, arg, sizeof(arg22)); \
}

#endif
