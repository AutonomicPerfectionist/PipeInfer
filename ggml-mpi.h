#pragma once
#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

struct ggml_context;
struct ggml_tensor;
struct ggml_cgraph;

#ifdef __cplusplus
extern "C" {
#endif

#define GGML_MPI_DECODE 0

#define GGML_MPI_KV_CLEAR 1

#define GGML_MPI_KV_SEQ_RM 2

#define GGML_MPI_KV_SEQ_CP 3

#define GGML_MPI_KV_SEQ_KEEP 4

#define GGML_MPI_KV_SEQ_SHIFT 5

#define GGML_MPI_SHUTDOWN 6

#define GGML_MPI_TRANSFER_TENSORS 7

#define GGML_MPI_SYNC_LOGITS 8

#define GGML_MPI_CANCEL_RUN 9

#define GGML_MPI_KV_SEQ_CP_BACK 10

#define GGML_MPI_TRANS_ID 11

#define GGML_MPI_BATCH_ID 12

#define GGML_MPI_N_TOKENS 13

#define GGML_MPI_TOKENS 14

#define GGML_MPI_N_SEQ_IDS 15

#define GGML_MPI_SEQ_IDS 16

#define GGML_MPI_POS 17

#define GGML_MPI_BEGIN_TRANSACTION 18

/**
 * The context used for MPI operations,
 * a program may make use of more than one
 * context but must always have at least one.
 *
 * The context stores required information like the
 * node rank and a communicator to use for MPI operations.
 * A context is guaranteed to be internally consistent,
 * meaning that a context's stored rank is valid within
 * the context's communicator.
 */
struct ggml_mpi_context;


int ggml_mpi_trans_id(struct ggml_mpi_context * ctx_mpi);

int ggml_mpi_recv_trans_id(struct ggml_mpi_context * ctx_mpi);

void ggml_mpi_inc_trans_id(struct ggml_mpi_context * ctx_mpi);

/**
 * Initialize the MPI library and the GGML MPI backend.
 * Calling more than once during the lifetime of the program
 * leads to undefined behavior. This function must be called before
 * any MPI operations.
 */
void ggml_mpi_backend_init(void);

bool ggml_mpi_is_decoding(struct ggml_mpi_context * ctx_mpi);

int ggml_mpi_status_count_int32(struct ggml_mpi_context * ctx_mpi);

void ggml_mpi_graph_creation_post(struct ggml_mpi_context * ctx_mpi, struct ggml_cgraph * cgraph, int   n_layers);

void ggml_mpi_wait_recv(struct ggml_mpi_context * ctx_mpi);

/**
 * Frees the MPI backend, must be called only once at termination
 * of the program. No MPI operations may be completed after calling this function,
 * and attempting to do so will lead to undefined behavior.
 */
void ggml_mpi_backend_free(void);

/**
 * Construct a new MPI context using the MPI_WORLD
 * communicator. This is useful only to create the
 * initial context, as calling multiple times
 * will only create effective copies of the same data.
 *
 * @return A context for us in the global communicator.
 */
struct ggml_mpi_context * ggml_mpi_init(void);

/**
 * Create a new context by splitting the given context's
 * communicator, creating a "sub-communicator." This is a collective
 * operation and must be performed by all nodes within the same communicator.
 * The color and key have the same meaning as in MPI_Comm_split(), i.e.
 * the color is used to determine the sub-communicator this node will belong to,
 * and the key is the relative rank of this node in the new communicator.
 *
 * An example: if a node passes a color of 1, and a different node passes a color of 2,
 * the nodes will belong to two different sub-communicators. If two nodes pass the same
 * color, then their ranks will be ordered by the order of their keys. If they pass the same
 * key, then the tie will be broken by the nodes' ranks in the old communicator.
 *
 * The communicator used by the given context remains entirely valid, so it is advisable
 * to store both the old and new contexts. This allows an application to
 * select at runtime which communicator to perform MPI operations with. An example
 * would be to segregate the nodes into multiple domains categorized by the functions
 * they perform, and use the original context to broadcast to all nodes in the cluster.
 *
 * @param ctx The context containing the communicator to split.
 * @param color The sub-communicator that this node will belong to.
 * @param key The relative rank of this node in the new communicator.
 * @return A new context with all values referencing the newly-created communicator.
 */
struct ggml_mpi_context * ggml_mpi_split_comm(struct ggml_mpi_context * ctx, int color, int key);

void ggml_mpi_barrier(struct ggml_mpi_context * ctx);

int ggml_mpi_next_node(struct ggml_mpi_context * ctx_mpi);

int ggml_mpi_prev_node(struct ggml_mpi_context * ctx_mpi);

void ggml_mpi_sync_ints_pipelined(
        struct ggml_mpi_context * ctx_mpi,
        int32_t * vals,
        int count,
        int tag
);

void ggml_mpi_sync_ints_pipelined_back(
        struct ggml_mpi_context * ctx_mpi,
        int32_t * vals,
        int count,
        int tag
);
// clear = 1, rm = 2, cp = 3, keep = 4, seq_shift = 5
void ggml_mpi_probe(struct ggml_mpi_context * ctx_mpi, int src, int tag);
int ggml_mpi_status_tag(struct ggml_mpi_context * ctx_mpi);

int ggml_mpi_iprobe(struct ggml_mpi_context * ctx_mpi, int src, int tag);

/**
 * Frees the given context, including the communicator. No MPI
 * operations besides ggml_mpi_backend_freee(void) should be executed after
 * running this function.
 *
 * @param ctx The context to free.
 */
void ggml_mpi_free(struct ggml_mpi_context * ctx);

/**
 * Get the rank of this node in the given context's communicator.
 *
 * @param ctx The context to use to determine the rank with regards to.
 * @return The rank of this node.
 */
int ggml_mpi_rank(struct ggml_mpi_context * ctx);

/**
 * Get the number of nodes that are a part of
 * the communicator referenced by the given context.
 *
 * @param ctx The context containing the communicator used for this size check.
 * @return The number of nodes that are a part of the given context's communicator.
 */
size_t ggml_mpi_size(struct ggml_mpi_context * ctx);

/**
 * Synchronize needed information among the nodes
 * to prepare for running an evaluation iteration.
 * This is a collective operation and all nodes must
 * call this function. It will block until all
 * nodes have entered it, to prevent any desync
 * between nodes.
 *
 * @param ctx_mpi The context in which to prepare for evaluation.
 * @param n_tokens A pointer to the n_tokens, which will be synchronized after this function.
 * @param pos A pointer to the pos array, which will be synchronized after this function.
 * @param n_seq_ids A pointer to the n_seq_ids array, which will be synchronized after this function.
 * @param seq_id A pointer to the seq_id 2D array, which will be synchronized after this function.
 * @param logits A pointer to the logits array, which is unused currently since only node 0 needs them.
 */
bool ggml_mpi_eval_init(
        struct ggml_mpi_context *   ctx_mpi,
                int32_t         *   n_tokens,
                int32_t         **  tokens,
                int32_t         **  pos,
                int32_t         **  n_seq_ids,
                int32_t         *** seq_id,
                int8_t          **  logits,
                int32_t         *   batch_id,
                bool                receive_only);

void ggml_mpi_synch_int(
        struct ggml_mpi_context     * ctx_mpi,
                int32_t * val,
                int root
        );

void ggml_mpi_synch_float(
        struct ggml_mpi_context     * ctx_mpi,
        float * val,
        int root
);

void ggml_mpi_recv_float_array(
        struct ggml_mpi_context     * ctx_mpi,
        float * val,
        int arr_size,
        int src,
        int tag
);

void ggml_mpi_send_float_array_async(
        struct ggml_mpi_context     * ctx_mpi,
        float * val,
        int arr_size,
        int dest,
        int tag
);

/**
 * Split a range across all nodes within the given
 * context, weighting the allocations by the given weights.
 * The dimensions of the returned 2d array are (number of nodes in the context, 2).
 * The first element in the inner array is the starting point of the range allocated
 * to the node indicated by the index into the outer array,
 * and the second element is the end point of the allocated range, inclusive.
 *
 * @param ctx_mpi The context used to determine the number of nodes
 *                to split the range across.
 * @param start The starting point of the range.
 * @param end The end point of the range, inclusive.
 * @param node_weights How to weight the allocations across the nodes,
 *                     must sum to 1.0.
 * @return A 2d array, the first dimension is the number of nodes in the context
 *         and the second dimension is 2.
 */
uint16_t** ggml_mpi_split_range(
    struct ggml_mpi_context * ctx_mpi,
    uint16_t start,
    uint16_t end,
    float node_weights[]
);

/**
 * Scatter the layer ranges across all nodes
 * in the given context. This is a collective operation
 * and must be called by all nodes that are within the same
 * communicator. The given layer ranges must be in the same
 * format as created by the ggml_mpi_split_range().
 *
 * @param ctx_mpi The context to scatter the layers across.
 * @param layer_ranges The pre-split ranges to scatter to the nodes.
 */
void ggml_mpi_scatter_layers(
    struct ggml_mpi_context * ctx_mpi,
    uint16_t ** layer_ranges
);

/**
 * Modify compute graph to only process allocated
 * layers.
 *
 * @param ctx_mpi The context containing the allocated layer range.
 * @param gf The compute graph to modify
 * @param n_layers The number of layers in the model, used as an upper bound in the layer ranges.
 */
bool ggml_mpi_graph_compute_pre(
        struct ggml_mpi_context * ctx_mpi,
             struct ggml_cgraph * gf);

/**
 * Sends the output tensor to the next node for processing
 * of later layers.
 *
 * @param ctx_mpi The context to use for MPI operations.
 * @param gf The graph used in the computations
 * @param n_layers The number of layers in the model.
 */
void ggml_mpi_graph_compute_post(
        struct ggml_mpi_context * ctx_mpi,
             struct ggml_cgraph * gf);

#ifdef __cplusplus
}
#endif
