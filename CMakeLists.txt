cmake_minimum_required( VERSION 2.8.1 )

project( MAGMA C CXX Fortran )


# ----------------------------------------
# to show compile commands, set this here or use 'make VERBOSE=1'
#set(CMAKE_VERBOSE_MAKEFILE on)


# ----------------------------------------
# don't regenerate files during make.
# (I think this means you have to manually re-run CMake if CMakeLists changes.
#  Hopefully it fixes the huge problems with CMake interrupting Visual Studio.)
set(CMAKE_SUPPRESS_REGENERATION on)


# ----------------------------------------
# force an out-of-source build, to not overwrite the existing Makefiles
# (out-of-source is cleaner, too)
string( COMPARE EQUAL "${CMAKE_SOURCE_DIR}" "${CMAKE_BINARY_DIR}" MAGMA_COMPILE_INPLACE )
if( MAGMA_COMPILE_INPLACE )
	message(FATAL_ERROR "Compiling MAGMA with CMake requires an out-of-source build. To proceed:
  rm -rf CMakeCache.txt CMakeFiles/   # delete files in ${CMAKE_SOURCE_DIR}
  mkdir build
  cd build
  cmake ..
  make" )
endif()


# ----------------------------------------
# check Fortran name mangling
include( FortranCInterface )
FortranCInterface_HEADER( ${CMAKE_SOURCE_DIR}/include/mangling.h MACRO_NAMESPACE MAGMA_ )


# ----------------------------------------
# locate OpenMP
find_package( OpenMP )
if ( OPENMP_FOUND )
	message( "-- Found OpenMP" )
	message( "   OpenMP_C_FLAGS   ${OpenMP_C_FLAGS}" )
	message( "   OpenMP_CXX_FLAGS ${OpenMP_CXX_FLAGS}" )
	set( CMAKE_C_FLAGS   "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}"   )
	set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}" )
endif()


# ----------------------------------------
# locate CUDA libraries
set( GPU_TARGET "Fermi Kepler" CACHE STRING "CUDA architectures to compile for; one or more of Tesla, Fermi, Kepler" )
find_package( CUDA )
if ( CUDA_FOUND )
	message( "-- Found CUDA ${CUDA_VERSION}" )
	message( "   CUDA_INCLUDE_DIRS:   ${CUDA_INCLUDE_DIRS}"   )
	message( "   CUDA_CUDART_LIBRARY: ${CUDA_CUDART_LIBRARY}" )
	#message( "   CUDA_LIBRARIES: ${CUDA_LIBRARIES}" )
	#message( "   CUDA_CUBLAS_LIBRARIES: ${CUDA_CUBLAS_LIBRARIES}" )
	include_directories( ${CUDA_INCLUDE_DIRS} )

	# NVCC options for the different cards
	# sm_xx is binary, compute_xx is PTX for forward compatability
	# MIN_ARCH is lowest requested version 
	# NV_SM    accumulates sm_xx for all requested versions
	# NV_COMP  is compute_xx for highest requested version
	set( NV_SM    "" )
	set( NV_COMP  "" )
	
	message( "   GPU_TARGET ${GPU_TARGET}" )
	if ( ${GPU_TARGET} MATCHES Tesla )
		if ( NOT MIN_ARCH )
			set( MIN_ARCH 130 )
		endif()
		# -gencode arch=compute_10,code=compute_10
		set( NV_SM "${NV_SM} -gencode arch=compute_13,code=sm_13" )
		set( NV_COMP        "-gencode arch=compute_13,code=compute_13" )
		message( "   compile for CUDA arch 1.3 (Tesla)" )
	endif()
	
	if ( ${GPU_TARGET} MATCHES Fermi )
		if ( NOT MIN_ARCH )
			set( MIN_ARCH 200 )
		endif()
		set( NV_SM "${NV_SM} -gencode arch=compute_20,code=sm_20" )
		set( NV_COMP        "-gencode arch=compute_20,code=compute_20" )
		message( "   compile for CUDA arch 2.x (Fermi)" )
	endif()
	
	if ( ${GPU_TARGET} MATCHES Kepler )
		if ( NOT MIN_ARCH )
			set( MIN_ARCH 300 )
		endif()
		set( NV_SM "${NV_SM} -gencode arch=compute_30,code=sm_30 -gencode arch=compute_35,code=sm_35" )
		set( NV_COMP        "-gencode arch=compute_35,code=compute_35" )
		message( "   compile for CUDA arch 3.x (Kepler)" )
	endif()
	
	if ( NOT MIN_ARCH )
		message( FATAL_ERROR "GPU_TARGET must contain one or more of Tesla, Fermi, Kepler." )
	endif()
	
	set( CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -DHAVE_CUBLAS ${NV_SM} ${NV_COMP}" )
	add_definitions( "-DHAVE_CUBLAS -DMIN_CUDA_ARCH=${MIN_ARCH}" )
endif()


# ----------------------------------------
# locate LAPACK libraries
set( LAPACK_LIBRARIES "" CACHE STRING "Libraries for LAPACK and BLAS, to manually override search" )
if ( "${LAPACK_LIBRARIES}" STREQUAL "" )
	message( "-- Searching for BLAS and LAPACK. To override, set LAPACK_LIBRARIES using ccmake." )
	find_package( LAPACK )
else()
	message( "-- User set LAPACK_LIBRARIES. To change, edit LAPACK_LIBRARIES using ccmake (set to empty to enable search)." )
endif()
message( "   BLAS_LIBRARIES:      ${BLAS_LIBRARIES}"      )
message( "   LAPACK_LIBRARIES:    ${LAPACK_LIBRARIES}"    )

# If using MKL, add it to includes and define MAGMA_WITH_MKL
# Initially, this gets MKLROOT from environment, but then the user can edit it.
set( MKLROOT $ENV{MKLROOT} CACHE STRING "MKL installation directory" )
if ( NOT "${MKLROOT}" STREQUAL "" )
	message( "-- MKLROOT set to ${MKLROOT}. To change, edit MKLROOT using ccmake." )
	include_directories( ${MKLROOT}/include )
	add_definitions( -DMAGMA_WITH_MKL )
else()
	message( "-- MKLROOT not set. To change, set MKLROOT using ccmake." )
endif()


# ----------------------------------------
# common flags
# On Windows:
#     Strip out /W3; we will use -W4
#     -Wall is way too verbose; use -W4
#     -MP enables parallel builds
#     -std=c99 is not implemented, so skip that
if ( WIN32 )
	string( REGEX REPLACE " */W3" "" CMAKE_C_FLAGS   "${CMAKE_C_FLAGS}"   )
	string( REGEX REPLACE " */W3" "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}" )
	set( CMAKE_C_FLAGS   "${CMAKE_C_FLAGS} -W4 -MP" )
	set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -W4 -MP" )
else()
	set( CMAKE_C_FLAGS   "${CMAKE_C_FLAGS} -Wall -std=c99" )
	set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall" )
endif()

if ( CMAKE_HOST_APPLE )
	# 64-bit veclib has issues, so compile 32-bit.
	# (I think an issue with prototypes return float instead of double)
	message( "-- MacOS X: setting to 32-bit, to avoid issues with 64-bit veclib" )
	set( CMAKE_C_FLAGS       "${CMAKE_C_FLAGS} -m32" )
	set( CMAKE_CXX_FLAGS     "${CMAKE_CXX_FLAGS} -m32" )
	set( CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} -m32" )
	set( CUDA_64_BIT_DEVICE_CODE OFF )
endif()

include_directories( include )
include_directories( control )

# Need to check sizeof(void*) after setting flags above;
# CMAKE_SIZEOF_VOID_P can be wrong.
include( CheckTypeSize )
CHECK_TYPE_SIZE( void* SIZEOF_VOID_PTR )
set( CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} -Dmagma_devptr_t=\"integer\(kind=${SIZEOF_VOID_PTR}\)\"" )

message( "-- Flags" )
message( "   CFLAGS   ${CMAKE_C_FLAGS}" )
message( "   CXXFLAGS ${CMAKE_CXX_FLAGS}" )
message( "   NFLAGS   ${CUDA_NVCC_FLAGS}" )
message( "   FFLAGS   ${CMAKE_Fortran_FLAGS}" )


# ----------------------------------------
# where to put libmagma
set( CMAKE_ARCHIVE_OUTPUT_DIRECTORY lib )
set( CMAKE_LIBRARY_OUTPUT_DIRECTORY lib )


# ----------------------------------------
# compile library
include( ${CMAKE_SOURCE_DIR}/src/CMake )
include( ${CMAKE_SOURCE_DIR}/control/CMake )
include( ${CMAKE_SOURCE_DIR}/magmablas/CMake )
include( ${CMAKE_SOURCE_DIR}/interface_cuda/CMake )

list( APPEND ALLSRC ${src_ALLSRC} ${control_ALLSRC} ${magmablas_ALLSRC} ${interface_cuda_ALLSRC} )

if ( WIN32 )
	# Windows seems to have a problem mixing C, CUDA, and Fortran files
	# Currently ignores .f90 and .F90 files, because it doesn't seem to
	# understand that .F90 files should be pre-processed.
	
	# separate Fortran and C/C++/CUDA files
	foreach( f ${ALLSRC} )
		if ( ${f} MATCHES "\\.(f)$" )  # |f90|F90
			list( APPEND ALLSRC_f   ${f} )
		elseif ( ${f} MATCHES "\\.(c|cu|cpp)$" )
			list( APPEND ALLSRC_cpp ${f} )
		endif()
	endforeach()
	#message( "ALLSRC_cpp ${ALLSRC_cpp}" )
	#message( "ALLSRC_f   ${ALLSRC_f}"   )
	
	# on Windows, Fortran files aren't compiled if listed here...
	cuda_add_library( magma ${ALLSRC_cpp} )
	target_link_libraries( magma
		${LAPACK_LIBRARIES}
		${CUDA_CUDART_LIBRARY}
		${CUDA_CUBLAS_LIBRARIES}
	)
	
	# no Fortran files at the moment (how to test ALLSRC_f is not empty?),
	# but keep this around for future.
	# ...so make a separate library out of Fortran files. Ugh.
	#add_library( magmaf ${ALLSRC_f} )
	#target_link_libraries( magmaf
	#	${LAPACK_LIBRARIES}
	#	${CUDA_CUDART_LIBRARY}
	#	${CUDA_CUBLAS_LIBRARIES}
	#)
	set( LIBS testing lapacktest magma )  #magmaf )
else()
	# Unix doesn't seem to have a problem with mixing C, CUDA, and Fortran files
	cuda_add_library( magma ${ALLSRC} )
	target_link_libraries( magma
		${LAPACK_LIBRARIES}
		${CUDA_CUDART_LIBRARY}
		${CUDA_CUBLAS_LIBRARIES}
	)
	set( LIBS testing lapacktest magma )
endif()


# ----------------------------------------
# compile testing library
set( TEST_SRC
	testing/magma_zutil.cpp
	testing/magma_cutil.cpp
	testing/magma_dutil.cpp
	testing/magma_sutil.cpp
	testing/magma_util.cpp
)
add_library( testing ${TEST_SRC} )


# ----------------------------------------
# compile lapacktest library
include( ${CMAKE_SOURCE_DIR}/testing/lin/CMake )
add_library( lapacktest ${testing_lin_ALLSRC} )


# ----------------------------------------
# compile each tester
set( CMAKE_RUNTIME_OUTPUT_DIRECTORY testing )
include( ${CMAKE_SOURCE_DIR}/testing/CMake )
list( REMOVE_ITEM testing_ALLSRC ${TEST_SRC} )
foreach( TEST ${testing_ALLSRC} )
	string( REGEX REPLACE "\\.(cpp|f90|F90)" "" EXE ${TEST} )
	string( REGEX REPLACE "testing/" "" EXE ${EXE} )
	#message( "${TEST} --> ${EXE}" )
	add_executable( ${EXE} ${TEST} )
	target_link_libraries( ${EXE} ${LIBS} )
endforeach()
