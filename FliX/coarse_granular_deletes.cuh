// =============================================================================
// File: coarse_granular_deletes.cuh
// Author: Rosina Kharal
// Description: Implements coarse_granular_deletes
// Copyright (c) 2025 Justus Henneberg, Rosina Kharal
// SPDX-License-Identifier: GPL-3.0-or-later
// =============================================================================

#ifndef INDEX_PROTOTYPE_IMPL_COARSE_GRANULAR_DELETES_CUH
#define INDEX_PROTOTYPE_IMPL_COARSE_GRANULAR_DELETES_CUH


#include <cstdint>
#include <cstdio>
#include "launch_parameters.cuh"
#include "definitions.cuh"
#include "definitions_updates.cuh"
#include "definitions_coarse_granular.cuh"
#include "coarse_granular_inserts.cuh"

//#define DEBUG_DELETION    //for debugging, print DELETION information and Shifting keys to Left '
//#define ERROR_CHECKS      //for debugging, check for errors, prints relevant error messages in INSERT of REMOVE


// Perform deletion by inserting a tombstone
template<typename key_type>
DEVICEQUALIFIER
void perform_delete_tombstone(void* curr_node, key_type delkey, smallsize key_index, smallsize num_elements, smallsize node_size, smallsize tid) {
   // int tid = blockIdx.x * blockDim.x + threadIdx.x;


    //if( delkey ==3879510667) DEBUG_PD_TB("Perform DELETE Tombstone: Tid:", tid, key_index, num_elements);


    // Set the tombstone at the specified key index
    set_key_node<key_type>(curr_node, key_index, static_cast<key_type>(tombstone));
    set_offset_node<key_type>(curr_node, key_index, static_cast<smallsize>(0));

#ifdef DEBUG_DELETION
    printf("After Insert Tombstone: Tid: %d, key_index %d, num_elements %d\n", tid, key_index, num_elements);
    print_node<key_type>(curr_node, node_size);
#endif
}


//peform all operations for deletions here
template<typename key_type>
DEVICEQUALIFIER
void perform_delete_shift(void* curr_node, smallsize key_index, smallsize num_elements, smallsize partition_size) {
    //smallsize tid = threadIdx.x;
    //smallsize idx = tid + key_index;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    //assert(key_index <= num_elements + 1 && key_index >= 2);
    //assert(key_index >= 1);

    if (key_index ==num_elements) {
        set_key_node<key_type>(curr_node, key_index, static_cast<key_type>(0));
        set_offset_node<key_type>(curr_node, key_index, static_cast<smallsize>(0));
        return;
    }   
	const smallsize num_shifts = (num_elements) - key_index; //need to also shift out th
    smallsize curr_index = key_index;
#ifdef DEBUG_DELETION
    printf("B4 Perform DEL Shift: Tid: %d, num_shifts %d, key_index %d, num_elements %d\n", tid, num_shifts, key_index, num_elements);
#endif
   
   
    for (smallsize i = 0; i < num_shifts; ++i) {
        // Extract key and offset from curr_node at index curr_index
        key_type curr_key = extract_key_node<key_type>(curr_node, curr_index+1);
        smallsize curr_offset = extract_offset_node<key_type>(curr_node, curr_index+1);
#ifdef DEBUG_DELETION
        printf("Tid: %d, key that is read and will move to one over to LEFT: curr_key %llu, curr offset: %u read from curr_index %d \n", tid, curr_key, curr_offset, curr_index);
#endif 
        // Set key and offset at index (curr_index - 1)
        set_key_node<key_type>(curr_node, (curr_index), curr_key);
        set_offset_node<key_type>(curr_node, (curr_index), curr_offset);

        // increment curr_index
        curr_index++;
    }

    //zero out num_elements;
    set_key_node<key_type>(curr_node, num_elements, static_cast<key_type>(0));
    set_offset_node<key_type>(curr_node, num_elements, static_cast<smallsize>(0));

#ifdef DEBUG_DELETION
    printf("After Delete Shift before Decrement Size: Tid: %d, num_shifts %d, key_index %d, num_elements %d PRINT TID = 0\n", tid, num_shifts, key_index, num_elements);
    print_node<key_type>(curr_node, partition_size);
#endif 
}


template <typename key_type>
//GLOBALQUALIFIER 
DEVICEQUALIFIER
void process_deletes(const key_type *__restrict__ update_list, smallsize update_size, key_type maxbucket, smallsize minindex, smallsize maxindex, updatable_cg_params* launch_params, void* curr_node, key_type num_elements, int tid) {
      //
      //compute tid here
    smallsize idx = blockIdx.x * blockDim.x + threadIdx.x;
	smallsize totalstoredsize = launch_params->stored_size;
	void* allocation_buffer = launch_params->allocation_buffer;
	smallsize partition_size = launch_params->partition_size;
    smallsize node_size = launch_params->node_size;
	smallsize partition_count_with_overflow = launch_params->partition_count_with_overflow;
	smallsize node_stride = launch_params->node_stride;
    smallsize allocation_buffer_count = launch_params->allocation_buffer_count;
	
	//smallsize returnval = 0;  //0 not inserted was present
							  //1 successfully inserted
							  //2 error of some type occured
   
    smallsize lastposition_bytes = get_lastposition_bytes<key_type>(node_size);
	auto buf = launch_params->ordered_node_pairs;
	smallsize maxkeys_pernode= launch_params->node_size; //allow 2xpartionsize keys per node 
																   //last index is used for next pointer
	

	//const key_type* update_list = static_cast<const key_type*>(launch_params->update_list);
    
#ifdef DEBUG_TID_0
    if (blockIdx.x == 0 && threadIdx.x == 0) {
       print_set_nodes<key_type>(launch_params, tid);
    }
    
    if (tid==0){
       //print out the update_list using update_list[1]/ the size of the list is 16
         printf("Tid: %d, PRINTING update/ Del LIST\n", tid);
         for (int i = 0; i < update_size; i++) {
              printf("Tid: %d, update_list[%d] %llu\n", tid, i, update_list[i]);
         }  
    }
#endif

   // --- printf(" Process Deletes: TID: %d, minindex %d, maxindex %d, update_size %d \n", tid, minindex, maxindex, launch_params->update_size);
    //const smallsize update_list_size = launch_params->update_size;

   //for (smallsize i = minindex; i <= maxindex; ++i) {
     for (smallsize i = minindex; i < update_size; ++i) {
        // Get the key from the update list
        //key_type key = static_cast<const key_type*>(launch_params->update_list)[i];
		key_type key = update_list[i];
        if (key > maxbucket) {
            break;
        }
        // Define insert index
        smallsize insert_index;
        bool found = false;
        //printf("ProcessUpdates: TID: %d, insert key %llu for i: %d :  minindex %d, maxindex %d \n", tid, key, i, minindex, maxindex);
        //printf("processUpdates: Tid: %d, minindex %d, maxindex %d\n", tid, minindex, maxindex);
        // Perform binary search
		//FOR LOOP HERE UNTIL WE FIND THE CORRECT CURR_NODE
		smallsize last_position_value = 0; //extract_key_node<key_type>(curr_node, lastpositionptr);
		key_type curr_max = cg::extract<key_type>(curr_node, 0);
		while (curr_max < key) {
			//printf("DEL: Travere Links tid:%d, curr_max %llu < is less than key %llu\n", tid, curr_max, key);	
			// Get the next node 
			last_position_value = cg::extract<smallsize>(curr_node, lastposition_bytes);
			last_position_value--; //decrement b/c it is inserted into node as +1 to avoid 0
#ifdef DEBUG_DELETION
			//NOTE: When we enter this code in error it is usually because list is unsorted
            //printf("DEL Traverse Link tid:%d, curr_max %llu is less than key %llu, last_position_value %llu \n", tid, curr_max, key, last_position_value);	
#endif
			if (last_position_value >= allocation_buffer_count) {
#ifdef ERROR_CHECKS
				printf("DEL ERROR: LAST PTR EXCEEDS NUM PARTITIONS tid %d, last_position_value: %d, node max: %llu \n", tid, last_position_value,curr_max);
#endif
				return;
			}
            //reinterpret_cast<uint8_t *>(allocation_buffer) + node_stride * free_value;
			curr_node = reinterpret_cast<uint8_t*>(allocation_buffer) + last_position_value * node_stride;
			curr_max = cg::extract<key_type>(curr_node, 0);
		}

#ifdef DEBUG_DELETION
        printf("DEL B4 BS Tid: %d minindex:%d, maxindex:%d -- Reached the correct node with key: %llu and max:  %llu and last_position_value %llu \n", idx, minindex,maxindex,key, curr_max, last_position_value);	
#endif 

        
		//printf("PU: B4 BS Tid: %d Reached the correct node with key: %llu and max:  %llu and last_position_value %llu \n", tid, key, curr_max, last_position_value);	
#ifdef DEBUG_TID_0
        if( tid ==0) {
            
            printf("PU: Tid: %d, Going to look for key: %llu in Binary Search in Cuda Buffer \n", tid, key);
            print_node<key_type>(curr_node, partition_size);

        }
#endif 


        //if (key == 2737346476){
        //        DEBUG_DEL_DEV("PD: Deleteion, Looking for ", tid, key, curr_max);
        //        print_node<key_type>(curr_node, partition_size*2);
        // }
    
        smallsize curr_size= cg::extract<smallsize>(curr_node, sizeof(key_type));
        if (curr_size > 0) {
            found = binary_search_in_cuda_buffer<key_type>(curr_node, curr_size, key, insert_index, tid);
        }

        if (!found) {
    #ifdef DEBUG_DELETION
                printf("PD: Deleteion, NOT FOUND: Tid: %d, key %llu Not present in node\n", tid, key);
                print_node<key_type>(curr_node, partition_size);
    #endif
                continue;  //move on to next key to insert
        }
        key_type key_at_index = extract_key_node<key_type>(curr_node, insert_index);
        //assert(key ==key_at_index);
        //printf("PUD going to Delete shift: Tid: %d, TRYING DELETE key: %llu \n", tid, key);
        perform_delete_shift<key_type>(curr_node, insert_index, curr_size,partition_size);
        //decrement size:
        smallsize prev_size = cg::extract<smallsize>(curr_node, sizeof(key_type));
        cg::set<smallsize>(curr_node, sizeof(key_type), prev_size - 1);
   
    }

#ifdef PRINT_DELETE_END
    __syncthreads();
    if (tid == 1)
    {
       printf("END DELETIONSS: PRINT NODES \n");
       //print_node<key_type>(curr_node, partition_size);
       print_set_nodes_and_links<key_type>(launch_params, tid);
    }
#endif
      
}
//}
//--------------- Delete with Tombstones
// Jan 2025



template <typename key_type>
//GLOBALQUALIFIER 
DEVICEQUALIFIER
void process_deletes_tombstones(key_type maxkey, smallsize minindex, smallsize maxindex, updatable_cg_params* launch_params, void* curr_node, key_type num_elements, int tid) {
      //
      //compute tid here
    smallsize idx = blockIdx.x * blockDim.x + threadIdx.x;
	//smallsize totalstoredsize = launch_params->stored_size;
	void* allocation_buffer = launch_params->allocation_buffer;
	smallsize partition_size = launch_params->partition_size;
    smallsize node_size = launch_params->node_size;
	smallsize partition_count_with_overflow = launch_params->partition_count_with_overflow;
	smallsize node_stride = launch_params->node_stride;
    smallsize allocation_buffer_count = launch_params->allocation_buffer_count;
   // const key_type *maxbuf = static_cast<const key_type *>(launch_params->maxvalues);


   
    smallsize lastposition_bytes = get_lastposition_bytes<key_type>(node_size);
	//auto buf = launch_params->ordered_node_pairs;
	//smallsize maxkeys_pernode= launch_params->node_size; //allow 2xpartionsize keys per node 
																   //last index is used for next pointer
	

	const key_type* update_list = static_cast<const key_type*>(launch_params->update_list);
    
  #ifdef DEBUG_TID_0
    if (blockIdx.x == 0 && threadIdx.x == 0) {
       print_set_nodes<key_type>(launch_params, tid);
    }
    if (tid==0){
       //print out the update_list using update_list[1]/ the size of the list is 16
         printf("Tid: %d, PRINTING update/ Del LIST\n", tid);
         for (int i = 0; i < launch_params->update_size; i++) {
              printf("Tid: %d, update_list[%d] %llu\n", tid, i, update_list[i]);
         }  
    }
  #endif

    //-----printf("Process Tomb Deletes: TID: %d, minindex %d, maxindex %d, update_size %d \n", tid, minindex, maxindex, launch_params->update_size);

    for (smallsize i = minindex; i <= maxindex; ++i) {
        // Get the key from the update list
        //key_type key = static_cast<const key_type*>(launch_params->update_list)[i];
		key_type key = update_list[i];
        // Define insert index
        smallsize insert_index;
        bool found = false;
        //printf("ProcessUpdates: TID: %d, insert key %llu for i: %d :  minindex %d, maxindex %d \n", tid, key, i, minindex, maxindex);
        //printf("processUpdates: Tid: %d, minindex %d, maxindex %d\n", tid, minindex, maxindex);
        // Perform binary search
		//FOR LOOP HERE UNTIL WE FIND THE CORRECT CURR_NODE
		smallsize last_position_value = 0; //extract_key_node<key_type>(curr_node, lastpositionptr);
		key_type curr_max = cg::extract<key_type>(curr_node, 0);


        //if( key ==3879510667) {   
        DEBUG_PD_TB("Before Loop of PTRS ", tid, key, curr_max);
            //print_node<key_type>(curr_node, partition_size);
       //}

		while (curr_max < key) {
			//printf("DEL: Travere Links tid:%d, curr_max %llu < is less than key %llu\n", tid, curr_max, key);	
			// Get the next node 
			last_position_value = cg::extract<smallsize>(curr_node, lastposition_bytes);
			last_position_value--; //decrement b/c it is inserted into node as +1 to avoid 0

			//NOTE: When we enter this code in error it is usually because list is unsorted
           //if( key ==3280486733) DEBUG_PD_TB("ENTERED DEL Traverse Link tid: ", curr_max, key, last_position_value);	

			if (last_position_value >= allocation_buffer_count) {
				DEBUG_PD_TB("DEL ERROR: LAST PTR EXCEEDS NUM PARTITIONS ",  tid, last_position_value, curr_max);
				return;
			}
            //reinterpret_cast<uint8_t *>(allocation_buffer) + node_stride * free_value;
			curr_node = reinterpret_cast<uint8_t*>(allocation_buffer) + last_position_value * node_stride;
			curr_max = cg::extract<key_type>(curr_node, 0);
		}

          smallsize curr_size= cg::extract<smallsize>(curr_node, sizeof(key_type));
        if (curr_size > 0) {
             //found = binary_search_in_cuda_buffer_with_tombstones<key_type>(curr_node, curr_size, key, insert_index, tid);
             found = linear_search_in_cuda_buffer_with_tombstones_full<key_type>(curr_node, curr_size, node_size, key, insert_index, tid);
        }

//#define DEBUG_DELETION_PRINT_NODE
        if (!found) {
    #ifdef DEBUG_DELETION_PRINT_NODE
    if( key ==912857181) {
                DEBUG_PD_TB("PD: Deleteion, NOT FOUND: Tid", tid, key);
                //print_node<key_type>(curr_node, partition_size);
    }
    #endif
                continue;  //move on to next key to insert
        }
       
       /// ERROR CHECK-->
       key_type key_at_index = extract_key_node<key_type>(curr_node, insert_index);

       //assert(key == key_at_index);

       // if( key ==3879510667) DEBUG_PD_TB("PUD going to DELETE WITH TOMBTONE: Tid: ", tid, key);

        perform_delete_tombstone<key_type>(curr_node, key, insert_index, curr_size, node_size, tid);

        // --->perform_delete_shift<key_type>(curr_node, insert_index, curr_size,partition_size);
        
        // Add in decrement size:
        smallsize prev_size = cg::extract<smallsize>(curr_node, sizeof(key_type));

#ifdef ERROR_CHECKS
         if (prev_size != curr_size) printf("DEL ERROR SIZE MIS MATCH: Tid: %d, prev_size %d, curr_size %d\n", tid, prev_size, curr_size);
#endif  
         //decrement size
         cg::set<smallsize>(curr_node, sizeof(key_type), prev_size - 1);
   
    }
#ifdef PRINT_DELETE_END
    __syncthreads();
    if (tid == 1)
    {
       printf("END DELETIONSS: PRINT NODES \n");
       //print_node<key_type>(curr_node, partition_size);
       print_set_nodes_and_links<key_type>(launch_params, tid);
    }
#endif
   

}
//}
#endif