#ifndef event_trace_h
#define event_trace_h

//////////////////////////////////////////////////////////////////////////////////////////////////
/* This section defines the macros for event logging */

#define DGETRF_COLOR 0xFF0000 	/* red */
#define DTSTRF_COLOR 0x800080   /* purple */
#define DGESSM_COLOR 0x0000FF	/* blue */
#define DSSSSM_COLOR 0x00FF00	/* green */
#define QUARK_COLOR 0xF5F5F5	/* black */
#define OVERHEAD1_COLOR 0x8A2BE2 /* purple1 */
#define OVERHEAD2_COLOR 0x4B0082 /* purple2 */

#define MAX_THREADS 60 
#define MAX_EVENTS 1048576
//#define MAX_EVENTS 16384
//define MAX_EVENTS 8192
//#define MAX_EVENTS 4096


#define core_event_start(my_core_id)			\
  event_start_time[my_core_id] = get_current_timee();

#define core_event_end(my_core_id)			\
  event_end_time[my_core_id] = get_current_timee();

#define core_log_event(event, my_core_id, tag)				\
  event_log[my_core_id][event_num[my_core_id]+0] = tag;			\
  event_log[my_core_id][event_num[my_core_id]+1] = event_start_time[my_core_id]; \
  event_log[my_core_id][event_num[my_core_id]+2] = event_end_time[my_core_id]; \
  event_log[my_core_id][event_num[my_core_id]+3] = (event);		\
  event_num[my_core_id] += (log_events << 2);				\
  event_num[my_core_id] &= (MAX_EVENTS-1);
//  if (event_num[my_core_id] >= MAX_EVENTS-5) event_num[my_core_id] -= (log_events << 2);

#define core_log_event_label(event, my_core_id, tag, label)              \
    event_label[my_core_id][event_num[my_core_id]+0] = (char *)strdup(label); \
    event_log[my_core_id][event_num[my_core_id]+0] = tag;               \
    event_log[my_core_id][event_num[my_core_id]+1] = event_start_time[my_core_id]; \
    event_log[my_core_id][event_num[my_core_id]+2] = event_end_time[my_core_id]; \
    event_log[my_core_id][event_num[my_core_id]+3] = (event);		\
    event_num[my_core_id] += (log_events << 2);				\
    event_num[my_core_id] &= (MAX_EVENTS-1);
//  if (event_num[my_core_id] >= MAX_EVENTS-5) event_num[my_core_id] -= (log_events << 2);


#endif /* event_trace_h */
