/*
 Enable some functions for debug
 *@author: Simplice Donfack
*/
#ifndef CA_DBG_TOOLS_H
#define CA_DBG_TOOLS_H

#ifdef MEMWATCH
#include "memwatch.h"
#endif

long ca_dbg_usecs ();

/*tracing macro*/
/*start trace*/
#define ca_trace_start() \
long tstart,tend;\
int tid;\
tstart =  ca_dbg_usecs();

/*end trace, cpu side*/
#define ca_trace_end_cpu(type) \
tend =  ca_dbg_usecs(); \
tid = schedule_Thread_Rank(sched_obj); \
ca_dbg_trace_add_event(tid, (type), tstart, tend, 0, 0, 0, 0);

/*end trace, gpu side*/
#define ca_trace_end_1gpu(type) \
ca_dbg_trace_device_sync(); \
tend =  ca_dbg_usecs(); \
tid = ca_dbg_trace_get_P()-1; \
ca_dbg_trace_add_event(tid, (type), tstart, tend, 0, 0, 0, 1);


/*init*/
void ca_dbg_trace_init(int P);
void ca_dbg_trace_add_event(int tid, char type, long tStart, long tEnd, int step, int col, int row, int stolen);
/*plot the trace and free memory*/
void ca_dbg_trace_finalize();
/*get the current number of threads for the tracing*/
int ca_dbg_trace_get_P();
/*Synchronize on the device only when need*/
void ca_dbg_trace_device_sync();


/*print a matrix*/
void ca_dbg_printMat(int M, int N, double *A,int LDA, char desc[] );

/*print the transpose of a matrix, M: number of columns of A (not transposed), N:number of rows*/
void ca_dbg_printMat_transpose(int M, int N, double *A,int LDA, char desc[] );

void ca_dbg_printMat_gpu(int M, int N, double *dA,int dA_LDA, char desc[] );

/*print the transpose of a matrix allocated on a device, M: number of columns of dA (not transposed), N:number of rows*/
void ca_dbg_printMat_transpose_gpu(int M, int N, double *dA,int dA_LDA, char desc[] );

/*write a matrix in a file*/
void ca_dbg_fwriteMat(char *filename, int M, int N, double *A, int LDA);

/*read a matrix from a file*/
void ca_dbg_freadMat(char *filename, int M, int N, double *A, int LDA);
#endif
