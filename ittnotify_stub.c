#ifdef __cplusplus
extern C {
#endif

typedef int iJIT_IsProfilingActiveFlags;
typedef int iJIT_JVM_EVENT;

unsigned int iJIT_GetNewMethodID(void) {
    return 0; // no profiling
}

iJIT_IsProfilingActiveFlags iJIT_IsProfilingActive(void) {
    return 0; // iJIT_NOTHING_RUNNING
}

int iJIT_NotifyEvent(iJIT_JVM_EVENT event_type, void *EventSpecificData) {
    (void)event_type;
    (void)EventSpecificData;
    return 0; // no-op
}

#ifdef __cplusplus
}
#endif
