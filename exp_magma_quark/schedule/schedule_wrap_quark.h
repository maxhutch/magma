/* 
    -- MAGMA (version 1.6.1) -- 
       Univ. of Tennessee, Knoxville 
       Univ. of California, Berkeley 
       Univ. of Colorado, Denver 
       Sept 2013 
 
       @author: Simplice Donfack 
*/
/* 
       This file is a wrapper for quark
       Actually, it uses quark as scheduler by default, but any scheduler can be used later by surcharging ALL of these functions.
*/
#ifndef SCHEDULE_WRAP_QUARK_H
#define SCHEDULE_WRAP_QUARK_H

#include "quark.h"

/*Quark constante*/
#define SCHEDULE_TASK_MAX_PRIORITY QUARK_TASK_MAX_PRIORITY
/*Quark structure*/
#define Schedule Quark
#define Schedule_Task_Flags Quark_Task_Flags
#define Schedule_Task_Flags_Initializer Quark_Task_Flags_Initializer

/*Quark subroutines to define*/
#define schedule_New            QUARK_New
#define schedule_Barrier        QUARK_Barrier
#define schedule_Delete            QUARK_Delete

#define schedule_Insert_Task    QUARK_Insert_Task
#define schedule_Task_Flag_Set    QUARK_Task_Flag_Set

#define schedule_Bit_Set        QUARK_Bit_Set
#define schedule_Thread_Rank    QUARK_Thread_Rank

#define schedule_unpack_args_1    quark_unpack_args_1
#define schedule_unpack_args_2    quark_unpack_args_2
#define schedule_unpack_args_3    quark_unpack_args_3
#define schedule_unpack_args_4    quark_unpack_args_4
#define schedule_unpack_args_5    quark_unpack_args_5
#define schedule_unpack_args_6    quark_unpack_args_6
#define schedule_unpack_args_7    quark_unpack_args_7
#define schedule_unpack_args_8    quark_unpack_args_8
#define schedule_unpack_args_9    quark_unpack_args_9
#define schedule_unpack_args_10    quark_unpack_args_10
#define schedule_unpack_args_11    quark_unpack_args_11
#define schedule_unpack_args_12    quark_unpack_args_12
#define schedule_unpack_args_13    quark_unpack_args_13
#define schedule_unpack_args_14    quark_unpack_args_14
#endif

