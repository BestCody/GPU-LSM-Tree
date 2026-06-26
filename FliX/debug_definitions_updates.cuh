// =============================================================================
// File: debug_definitions_updates.cuh
// Author: Rosina Kharal
// Description: Implements debug_definitions_updates     
//             Debugging macros for updates in FliX, used for testing and debugging the implementation of updates, lookups, and rebuilds in FliX. 
//             These macros can be enabled or disabled by defining or undefining the corresponding preprocessor directives at the top of the file.
//             This print statements show output in the codebase for host or device side debugging, and are used for testing and debugging.  
// Copyright (c) 2025 Justus Henneberg, Rosina Kharal
// SPDX-License-Identifier: GPL-3.0-or-later
// =============================================================================

#ifndef DEBUG_DEFINITIONS_UPDATES_H
#define DEBUG_DEFINITIONS_UPDATES_H

// Options to Print out Entire set of keys.
//#define PRINT_MAXVALUES
//#define PRINT_BUILD_VALUES
//#define PRINT_UPDATE_VALUES
//#define PRINT_GENERATED_KEYS
//#define PRINT_INSERT_VALUES
//#define PRINT_REMOVE_VALUES
//#define PRINT_LOOKUP_VALUES
//#define PRINT_PROCESS_INSERTS
//#define PRINT_PROCESS_INSERTS_DEBUG
//#define PRINT_PROCESS_INSERTS_END
//#define PRINT_REBUILD_DATA_END
#define PRINT_REBUILD_MERGE_DATA
//#define PRINT_PROCESS_DELETES_END
//----#define DEBUG_PRINT_REUSE_LIST
//#define PRINT_LOOKUPS_END
//#define PRINT_GENERATED_KEYS
//------PRINT_REMOVE_VALUES
//#define PRINT_LOOKUP_VALUES
//-----#define PRINT_SUCCESSOR_VALUES
//#define DBG_MISS2_CERR
//#define DEBUG_SUCCESSOR_RESULTS
//#define LSM_DEBUG_SUCCESSOR_KERNEL_PRINT_TREE
//#define LSM_DEBUG_DELETE_KERNEL_PRINT_TREE
// SECOND APPROACH WITH NO INDEX LAYER and SORTED PROBES


//#define DEBUG_MISS2
#ifdef DEBUG_MISS2
#define DBG_MISS2_CERR(expr)            \
    do                                  \
    {                                   \
        std::cerr << expr << std::endl; \
    } while (0)
#else
#define DBG_MISS2_CERR(expr) \
    do                       \
    {                        \
    } while (0)
#endif

// MAX 3 Args for Device Side Debugging is enabled
#define PRINT_DEVICE(msg, ...) \
    PRINT_DEVICE_IMPL(msg, ##__VA_ARGS__, 0, 0) // zero for default vals, max 3

#define PRINT_DEVICE_IMPL(msg, arg1, arg2, arg3, ...) \
    do { \
        printf("%s: " #arg1 " = %llu, " #arg2 " = %llu, " #arg3 " = %llu\n", \
               msg, \
               static_cast<unsigned long long>(arg1), \
               static_cast<unsigned long long>(arg2), \
               static_cast<unsigned long long>(arg3)); \
    } while (0)

#define PRINT_VARS(msg, count, ...) \
    do { \
        printf("%s: ", msg); \
        print_vars(#__VA_ARGS__, __VA_ARGS__); \
    } while (0)

__device__ __host__ const char* custom_strchr(const char* str, char ch) {
    // Custom strchr function that works for both host and device
    while (*str) {
        if (*str == ch) return str;
        ++str;
    }
    return nullptr;
}

__device__ __host__ inline void print_vars(const char* names) {
    // Base case: if there are no arguments, just print a newline
    printf("\n");
}

template<typename... Args>
__device__ __host__ inline void print_vars(const char* names, int count, Args... args) {
    print_vars(names, args...);
}

template<typename T, typename... Args>
__device__ __host__ inline void print_vars(const char* names, T first, Args... args) {
    const char* comma = custom_strchr(names, ',');
    if (comma) {
        printf("%.*s = %llu, ", (int)(comma - names), names, static_cast<unsigned long long>(first));
        print_vars(comma + 1, args...);
    } else {
        //printf("%s = %llu\n", names, static_cast<unsigned long long>(first));
        printf("%s = %u\n", names, static_cast<unsigned long long>(first));
    }
}

//--------- ENABLE VARYING LEVELS OF DEBUG STATEMENTS ----------------


// Get Data Structure Statistics 
//#define ENABLE_DEBUG_SUCCESSOR
#ifdef ENABLE_DEBUG_SUCCESSOR
    #define DEBUG_SUCCESSOR(msg, count, ...) PRINT_VARS(msg, count, __VA_ARGS__)
#else
    #define DEBUG_SUCCESSOR(msg, count, ...) // Empty
#endif

// Get Data Structure Statistics 
//#define ENABLE_ERROR_CHECK_BUILD
#ifdef ENABLE_DATA_STRUCTURE_STATS
    #define DATA_STRUCTURE_STATS(msg, count, ...) PRINT_VARS(msg, count, __VA_ARGS__)
#else
    #define DATA_STRUCTURE_STATS(msg, count, ...) // Empty
#endif

// Build Related Errors  in IMPL_CG_RTX_INDEX_UPDATES
//#define ENABLE_ERROR_CHECK_BUILD
#ifdef ENABLE_ERROR_CHECK_BUILD
    #define ERROR_CHECK_BUILD(msg, count, ...) PRINT_VARS(msg, count, __VA_ARGS__)
#else
    #define ERROR_CHECK_BUILD(msg, count, ...) // Empty
#endif

// DEBUG the Initial or Subsequent BUILDs of the Data structure in IMPL_CG_RTX_INDEX_UPDATES
//#define ENABLE_DEBUG_BUILD
#ifdef ENABLE_DEBUG_BUILD
    #define DEBUG_BUILD(msg, count, ...) PRINT_VARS(msg, count, __VA_ARGS__)
#else
    #define DEBUG_BUILD(msg, count, ...) // Empty
#endif

// DEBUG the Initial or Subsequent BUILDs of the Data structure in IMPL_CG_RTX_INDEX_UPDATES
//#define ENABLE_DEBUG_REBUILD
#ifdef ENABLE_DEBUG_REBUILD
    #define DEBUG_REBUILD(msg, count, ...) PRINT_VARS(msg, count, __VA_ARGS__)
#else
    #define DEBUG_REBUILD(msg, count, ...) // Empty
#endif


// DEBUG the Initial or Subsequent BUILDs of the Data structure in IMPL_CG_RTX_INDEX_UPDATES
#define ENABLE_DEBUG_BUILD_PARAMS
#ifdef ENABLE_DEBUG_BUILD_PARAMS
    #define DEBUG_BUILD_PARAMS(msg, count, ...) PRINT_VARS(msg, count, __VA_ARGS__)
#else
    #define DEBUG_BUILD_PARAMS(msg, count, ...) // Empty
#endif

// DEBUG the Initial or Subsequent BUILDs of the Data structure in IMPL_CG_RTX_INDEX_UPDATES
//#define ENABLE_DEBUG_GPU_RES_BYTES_PARAMS
#ifdef ENABLE_DEBUG_GPU_RES_BYTES_PARAMS
    #define DEBUG_GPU_RESIDENT_BYTES(msg, count, ...) PRINT_VARS(msg, count, __VA_ARGS__)
#else
    #define DEBUG_GPU_RESIDENT_BYTES(msg, count, ...) // Empty
#endif

// DEBUG Extra Number of Nodes Cacl in IMPL_CG_RTX_INDEX_UPDATES
//#define ENABLE_DEBUG_EXTRA_NODE_CALC
#ifdef ENABLE_DEBUG_EXTRA_NODE_CALC
    #define DEBUG_EXTRA_NODES(msg, count, ...) PRINT_VARS(msg, count, __VA_ARGS__)
#else
    #define DEBUG_EXTRA_NODES(msg, count, ...) // Empty
#endif

// DEBUG UPDATES in IMPL_CG_RTX_INDEX_UPDATES
//#define ENABLE_DEBUG_UPDATES
#ifdef ENABLE_DEBUG_UPDATES
    #define DEBUG_UPDATES(msg, count, ...) PRINT_VARS(msg, count, __VA_ARGS__)
#else
    #define DEBUG_UPDATES(msg, count, ...) // Empty
#endif

// DEBUG LOOKUPS in IMPL_CG_RTX_INDEX_UPDATES
//#define ENABLE_DEBUG_LOOKUPS
#ifdef ENABLE_DEBUG_LOOKUPS
    #define DEBUG_LOOKUPS(msg, count, ...) PRINT_VARS(msg, count, __VA_ARGS__)
#else
    #define DEBUG_LOOKUPS(msg, count, ...) // Empty
#endif

//-------------------- Device-side debugging -----------------------------
//#define ENABLE_DEBUG_UPDATES_DEVICE

#ifdef ENABLE_DEBUG_UPDATES_DEVICE
    #define DEBUG_UPDATES_DEVICE(msg, ...) PRINT_DEVICE(msg, ##__VA_ARGS__)
#else
    #define DEBUG_UPDATES_DEVICE(msg, ...) // Empty
#endif

// DEBUG MAXVALUES in IMPL_CG_RTX_INDEX_UPDATES
//#define ENABLE_DEBUG_MAXVALUES //note this is device side
#ifdef ENABLE_DEBUG_MAXVALUES
    #define DEBUG_MAXVALUES(msg, ...) PRINT_DEVICE(msg, ##__VA_ARGS__)
#else
    #define DEBUG_MAXVALUES(msg, ...) // Empty
#endif

//--------------------- Device-side debugging ** Inserts ** ---------------------//
//#define ENABLE_ERROR_INSERTS_DEVICE
#ifdef ENABLE_ERROR_INSERTS_DEVICE
    #define ERROR_INSERTS(msg, ...) PRINT_DEVICE(msg, ##__VA_ARGS__)
#else
    #define ERROR_INSERTS(msg, ...) // Empty
#endif

//#define ENABLE_DEBUG_PROCESS_INSERTS
#ifdef ENABLE_DEBUG_PROCESS_INSERTS
    #define DEBUG_PI(msg, ...) PRINT_DEVICE(msg, ##__VA_ARGS__)
#else
    #define DEBUG_PI(msg, ...) // Empty
#endif

//DEBUG_PI_BUCKET_NUMINSERTS
//#define ENABLE_DEBUG_PROCESS_INSERTS_BUCKET_NUMINSERTS
#ifdef ENABLE_DEBUG_PROCESS_INSERTS_BUCKET_NUMINSERTS
    #define DEBUG_PI_BUCKET_NUMINSERTS(msg, ...) PRINT_DEVICE(msg, ##__VA_ARGS__)
#else
    #define DEBUG_PI_BUCKET_NUMINSERTS(msg, ...) // Empty
#endif

//#define ENABLE_DEBUG_PROCESS_INSERTS_BUCKET
#ifdef ENABLE_DEBUG_PROCESS_INSERTS_BUCKET
    #define DEBUG_PI_BUCKET(msg, ...) PRINT_DEVICE(msg, ##__VA_ARGS__)
#else
    #define DEBUG_PI_BUCKET(msg, ...) // Empty
#endif

//#define ENABLE_DEBUG_PROCESS_INSERTS_BULK
#ifdef ENABLE_DEBUG_PROCESS_INSERTS_BULK
    #define DEBUG_PI_BULK(msg, ...) PRINT_DEVICE(msg, ##__VA_ARGS__)
#else
    #define DEBUG_PI_BULK(msg, ...) // Empty
#endif

//#define ENABLE_DEBUG_PROCESS_INSERTS_BUCKET_SPLIT
#ifdef ENABLE_DEBUG_PROCESS_INSERTS_BUCKET_SPLIT
    #define DEBUG_PI_BUCKET_SPLIT(msg, ...) PRINT_DEVICE(msg, ##__VA_ARGS__)
#else
    #define DEBUG_PI_BUCKET_SPLIT(msg, ...) // Empty
#endif

//#define ENABLE_DEBUG_PROCESS_INSERTS_TILE
#ifdef ENABLE_DEBUG_PROCESS_INSERTS_TILE
    #define DEBUG_PI_TILE(msg, ...) PRINT_DEVICE(msg, ##__VA_ARGS__)
#else
    #define DEBUG_PI_TILE(msg, ...) // Empty
#endif


//#define ENABLE_DEBUG_PROCESS_INSERTS_TILE_BULK
#ifdef ENABLE_DEBUG_PROCESS_INSERTS_TILE_BULK
    #define DEBUG_PI_TILE_BULK(msg, ...) PRINT_DEVICE(msg, ##__VA_ARGS__)
#else
    #define DEBUG_PI_TILE_BULK(msg, ...) // Empty
#endif

//--------------------- Device-side debugging ** Deletes **

//#define ENABLE_DEBUG_PROCESS_DELETES_TOMBSTONE
#ifdef ENABLE_DEBUG_PROCESS_DELETES_TOMBSTONE
    #define DEBUG_PD_TB(msg, ...) PRINT_DEVICE(msg, ##__VA_ARGS__)
#else
    #define DEBUG_PD_TB(msg, ...) // Empty
#endif

//#define ENABLE_DEBUG_PROCESS_DELETES_NODES
#ifdef ENABLE_DEBUG_PROCESS_DELETES_NODES
    #define DEBUG_DEL_DEV(msg, ...) PRINT_DEVICE(msg, ##__VA_ARGS__)
#else
    #define DEBUG_DEL_DEV(msg, ...) // Empty
#endif

//#define ENABLE_DEBUG_PROCESS_DELS_TILE
#ifdef ENABLE_DEBUG_PROCESS_DELS_TILE
    #define DEBUG_PI_TILE_DELS(msg, ...) PRINT_DEVICE(msg, ##__VA_ARGS__)
#else
    #define DEBUG_PI_TILE_DELS(msg, ...) // Empty
#endif

//--------------------- Device-side debugging ** LOOKUPS **

//#define ENABLE_DEBUG_PROCESS_LOOKUPS_TOMBSTONE
#ifdef ENABLE_DEBUG_PROCESS_LOOKUPS_TOMBSTONE
    #define DEBUB_LOOKUP_LINEAR_SEARCH(msg, ...) PRINT_DEVICE(msg, ##__VA_ARGS__)
#else
    #define DEBUB_LOOKUP_LINEAR_SEARCH(msg, ...) // Empty
#endif

//#define ENABLE_DEBUG_LOOKUP_NODES
#ifdef ENABLE_DEBUG_LOOKUP_NODES
    #define DEBUG_LOOKUP_NODES(msg, ...) PRINT_DEVICE(msg, ##__VA_ARGS__)
#else
    #define DEBUG_LOOKUP_NODES(msg, ...) // Empty
#endif

//#define ENABLE_DEBUG_PROCESS_LOOKUPS
#ifdef ENABLE_DEBUG_PROCESS_LOOKUPS
    #define DEBUB_LOOKUP(msg, ...) PRINT_DEVICE(msg, ##__VA_ARGS__)
#else
    #define DEBUB_LOOKUP(msg, ...) // Empty
#endif

//#define ENABLE_DEBUG_LOOKUPS_TOMBSTONE
#ifdef ENABLE_DEBUG_LOOKUPS_TOMBSTONE
    #define DEBUB_LOOKUP_TOMBSTONES(msg, ...) PRINT_DEVICE(msg, ##__VA_ARGS__)
#else
    #define DEBUB_LOOKUP_TOMBSTONES(msg, ...) // Empty
#endif

//#define ENABLE_DEBUG_LOOKUPS_DEV
#ifdef ENABLE_DEBUG_LOOKUPS_DEV
    #define DEBUG_LOOKUP_DEV(msg, ...) PRINT_DEVICE(msg, ##__VA_ARGS__)
#else
    #define DEBUG_LOOKUP_DEV(msg, ...) // Empty
#endif

//--------------------- Device-side debugging ** REBUILD **
#define ENABLE_DEBUG_REBUILD_DEV
#ifdef ENABLE_DEBUG_REBUILD_DEV
    #define DEBUG_REBUILD_DEV(msg, ...) PRINT_DEVICE(msg, ##__VA_ARGS__)
#else
    #define DEBUG_REBUILD_DEV(msg, ...) // Empty
#endif

//--------------------- Device-side debugging ** COMBINED **
//#define ENABLE_DEBUG_COMBINED
#ifdef ENABLE_DEBUG_COMBINED
    #define DEBUG_COMBINED(msg, ...) PRINT_DEVICE(msg, ##__VA_ARGS__)
#else
    #define DEBUG_COMBINED(msg, ...) // Empty
#endif
//--------------------- Device-side debugging ** COMPUTE_TOTAL_SIZE **

//#define ENABLE_DEBUG_COMPUTE_TOTAL
#ifdef ENABLE_DEBUG_COMPUTE_TOTAL
    #define DEBUG_COMPUTE_TOTAL(msg, ...) PRINT_DEVICE(msg, ##__VA_ARGS__)
#else
    #define DEBUG_COMPUTE_TOTAL(msg, ...) // Empty
#endif

#endif // DEBUG_DEFINITIONS_UPDATES_CUH