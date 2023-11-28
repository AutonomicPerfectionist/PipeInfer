#include "ggml-mpi.h"

#include "ggml.h"

#include <mpi.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MIN(a, b) ((a) < (b) ? (a) : (b))

#define UNUSED GGML_UNUSED

struct ggml_mpi_context {
    int rank;
    int size;
    MPI_Comm comm;
    int layer_start;
    int layer_end;
    MPI_Status status;
    MPI_Request asyncSendRequest;
    struct ggml_tensor * duped_send_tensor;
    MPI_Request asyncRecvRequest;
    struct ggml_tensor * duped_recv_tensor;
    bool asyncSendWaiting;
    bool asyncRecvWaiting;
    struct ggml_cgraph * cgraph;
    bool async;
    bool running_decode;
    bool res;
    bool embed;
    void* send_buffer;
};

void ggml_mpi_sync_pipelined(
        struct ggml_mpi_context *   ctx_mpi,
        void * val,
        int count,
        MPI_Datatype datatype,
        int tag
);

void ggml_mpi_backend_init(void) {
    int ret;
    MPI_Init_thread(NULL, NULL, MPI_THREAD_MULTIPLE, &ret);
}

void ggml_mpi_backend_free(void) {
    MPI_Finalize();
}

struct ggml_mpi_context * ggml_mpi_init(void) {
    struct ggml_mpi_context * ctx = calloc(1, sizeof(struct ggml_mpi_context));

    MPI_Comm_rank(MPI_COMM_WORLD, &ctx->rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ctx->size);
    ctx->comm = MPI_COMM_WORLD;
    ctx->asyncSendWaiting = false;
    ctx->asyncRecvWaiting = false;
    ctx->running_decode = false;
    ctx->async = false;
    ctx->send_buffer = calloc(1, 128*1024*1024); // 128MB buffer
    MPI_Buffer_attach(ctx->send_buffer, 4096*1024*32);

    return ctx;
}

struct ggml_mpi_context * ggml_mpi_split_comm(struct ggml_mpi_context * ctx, int color, int key) {
    if (color < 0) {
        color = MPI_UNDEFINED;
    }
    struct ggml_mpi_context * newCtx = calloc(1, sizeof(struct ggml_mpi_context));
    MPI_Comm_split(ctx->comm, color, key, &newCtx->comm);
    if(newCtx->comm == MPI_COMM_NULL) {
        newCtx->rank = -1;
        newCtx->size = -1;
        return newCtx;
    }
    MPI_Comm_rank(newCtx->comm, &newCtx->rank);
    MPI_Comm_size(newCtx->comm, &newCtx->size);
    return newCtx;
}

void ggml_mpi_free(struct ggml_mpi_context * ctx) {
    if(ctx->comm == MPI_COMM_NULL) {
        return;
    }
    ggml_mpi_sync_pipelined(ctx, NULL, 0, MPI_INT8_T, 6);
    MPI_Comm_free(&(ctx->comm));
    free(ctx);
}

bool ggml_mpi_is_decoding(struct ggml_mpi_context * ctx_mpi) {
    return ctx_mpi->running_decode;
}

struct ggml_cgraph * ggml_mpi_get_cgraph(struct ggml_mpi_context * ctx_mpi) {
    return ctx_mpi->cgraph;
}

void ggml_mpi_set_cgraph(struct ggml_mpi_context * ctx_mpi, struct ggml_cgraph * cgraph) {
    ctx_mpi->cgraph = cgraph;
}

int ggml_mpi_rank(struct ggml_mpi_context * ctx) {
    return ctx->rank;
}

size_t ggml_mpi_size(struct ggml_mpi_context * ctx) {
    return ctx->size;
}

void ggml_mpi_barrier(struct ggml_mpi_context * ctx_mpi) {
    MPI_Barrier(ctx_mpi->comm);
}

void ggml_mpi_probe(struct ggml_mpi_context * ctx_mpi, int src, int tag) {
    MPI_Probe((src >= 0) ? src : MPI_ANY_SOURCE, (tag >= 0) ? tag : MPI_ANY_TAG, ctx_mpi->comm, &(ctx_mpi->status));
}

int ggml_mpi_status_tag(struct ggml_mpi_context * ctx_mpi) {
    return ctx_mpi->status.MPI_TAG;
}

int ggml_mpi_next_node(struct ggml_mpi_context * ctx_mpi) {
    return (ctx_mpi->rank + 1) % ctx_mpi->size;
}

int ggml_mpi_prev_node(struct ggml_mpi_context * ctx_mpi) {
    int temp = (ctx_mpi->rank - 1);
    return (temp >= 0) ? temp : ctx_mpi->size - 1;
}

void ggml_mpi_sync_pipelined_recv(
        struct ggml_mpi_context *   ctx_mpi,
        void * val,
        int count,
        MPI_Datatype datatype,
        int tag
) {
    if(ctx_mpi->comm == MPI_COMM_NULL) {
        return;
    }
    MPI_Recv(val, count, datatype, ggml_mpi_prev_node(ctx_mpi), tag, ctx_mpi->comm, MPI_STATUS_IGNORE);

}


void ggml_mpi_sync_pipelined(
        struct ggml_mpi_context *   ctx_mpi,
        void * val,
        int count,
        MPI_Datatype datatype,
        int tag
        ) {
    if(ctx_mpi->comm == MPI_COMM_NULL) {
        return;
    }

    //printf("Rank %d sync pipelined\n", ctx_mpi->rank);


    if (ctx_mpi->rank != 0) {
        MPI_Recv(val, count, datatype, ggml_mpi_prev_node(ctx_mpi), tag, ctx_mpi->comm, MPI_STATUS_IGNORE);
    }
    if(ctx_mpi->rank < ctx_mpi->size - 1) {
        const int retval = MPI_Bsend(val, count, datatype, ggml_mpi_next_node(ctx_mpi), tag, ctx_mpi->comm);
        GGML_ASSERT(retval == MPI_SUCCESS);

    }
}

bool ggml_mpi_eval_init(
        struct ggml_mpi_context *   ctx_mpi,
                int32_t         *   n_tokens,
                int32_t         **  tokens,
                int32_t         **  pos,
                int32_t         **  n_seq_ids,
                int32_t         *** seq_id,
                int8_t          **  logits,
                bool                receive_only) {
    if(ctx_mpi->comm == MPI_COMM_NULL) {
        return false;
    }
    int32_t old_n_tokens = *n_tokens;


    ggml_mpi_sync_pipelined(ctx_mpi, n_tokens, 1, MPI_INT, 0);


    // For now, we assume that the pos, seq_ids, tokens, etc have been
    // pre-allocated for the largest possible sizes, even on worker nodes.
    //if (old_n_tokens != *n_tokens) {
    //    *pos = realloc(*pos, *n_tokens * sizeof(int32_t));
    //    *n_seq_ids = realloc(*n_seq_ids, *n_tokens * sizeof(int32_t ));
    //    *tokens = realloc(*tokens, *n_tokens * sizeof(int32_t ));
    //}

    ggml_mpi_sync_pipelined(ctx_mpi, *tokens, *n_tokens, MPI_INT32_T, 0);


    ggml_mpi_sync_pipelined(ctx_mpi, *n_seq_ids, *n_tokens, MPI_INT32_T, 0);

    // We need to know the total number of sequence
    // ids, so we count them all up
    int32_t total_n_seq_ids = 0;
    for (int32_t i = 0; i < *n_tokens; i++) {
        total_n_seq_ids += (*n_seq_ids)[i];
    }

    // MPI can't chase the pointers for multidimensional arrays, so we flatten them first
    // for transit
    int32_t * flattened_seq_ids = calloc(total_n_seq_ids, sizeof(int32_t));

    int32_t current_index = 0;

    // Only rank 0 needs to flatten since the others don't have the real seq_id
    if (ctx_mpi->rank == 0) {
        for (int32_t i = 0; i < *n_tokens; i++) {
            for (int32_t j = 0; j < (*n_seq_ids)[i]; j++) {
                flattened_seq_ids[current_index] = (*seq_id)[i][j];
                current_index++;
            }
        }
    }



    ggml_mpi_sync_pipelined(ctx_mpi, *pos, *n_tokens, MPI_INT32_T, 0);
    ggml_mpi_sync_pipelined(ctx_mpi, flattened_seq_ids, total_n_seq_ids, MPI_INT32_T, 0);

    current_index = 0;
    for (int32_t i = 0; i < *n_tokens; i++) {
        for (int32_t j = 0; j < (*n_seq_ids)[i]; j++) {
            (*seq_id)[i][j] = flattened_seq_ids[current_index];
            current_index++;
        }

    }
    free(flattened_seq_ids);

    return true;
}

void ggml_mpi_sync_ints_pipelined(
        struct ggml_mpi_context * ctx_mpi,
        int32_t * vals,
        int count,
        int tag
) {
    ggml_mpi_sync_pipelined(ctx_mpi, vals, count, MPI_INT32_T, tag);
}

void ggml_mpi_synch_int(
        struct ggml_mpi_context * ctx_mpi,
                        int32_t * val,
                        int root
) {
    if(ctx_mpi->comm == MPI_COMM_NULL) {
        return;
    }
//    printf("Rank %d sync int\n", ctx_mpi->rank);
    MPI_Bcast(val, 1, MPI_INT32_T, root, ctx_mpi->comm);
}

void ggml_mpi_synch_float(
        struct ggml_mpi_context * ctx_mpi,
        float * val,
        int root
) {
    if(ctx_mpi->comm == MPI_COMM_NULL) {
        return;
    }
//    printf("Rank %d sync float\n", ctx_mpi->rank);
    MPI_Bcast(val, 1, MPI_FLOAT, root, ctx_mpi->comm);
}

void ggml_mpi_recv_float_array(
        struct ggml_mpi_context     * ctx_mpi,
        float * val,
        int arr_size,
        int src,
        int tag
) {
//    printf("Rank %d recv float array, count=%d\n", ctx_mpi->rank, arr_size);
    int ret = MPI_Recv(val, arr_size, MPI_FLOAT, src, tag, ctx_mpi->comm, MPI_STATUS_IGNORE);
    GGML_ASSERT(ret == MPI_SUCCESS);
}

void ggml_mpi_send_float_array_async(
        struct ggml_mpi_context     * ctx_mpi,
        float * val,
        int arr_size,
        int dest,
        int tag
) {
//    printf("Rank %d send float array async, count=%d\n", ctx_mpi->rank, arr_size);
    int ret = MPI_Bsend(val, arr_size, MPI_FLOAT, dest, tag, ctx_mpi->comm);
    GGML_ASSERT(ret == MPI_SUCCESS);
}

static int ggml_graph_get_node_idx(struct ggml_cgraph * gf, const char * name) {
    struct ggml_tensor * t = ggml_graph_get_tensor(gf, name);
    if (t == NULL) {
        fprintf(stderr, "%s: tensor %s not found\n", __func__, name);
        return -1;
    }

    for (int i = 0; i < gf->n_nodes; i++) {
        if (gf->nodes[i] == t) {
            return i;
        }
    }

    fprintf(stderr, "%s: tensor %s not found in graph (should not happen)\n", __func__, name);
    return -1;
}

struct ggml_tensor * ggml_mpi_dup_tensor(struct ggml_tensor * t) {
    struct ggml_tensor * duped = malloc(sizeof(struct ggml_tensor));
    for (int i = 0; i < 4; i++) {
        duped->ne[i] = t->ne[i];
    }
    size_t data_size = ggml_element_size(t) * ggml_nelements(t);
    duped->data = malloc(data_size);
    memcpy(duped->data, t->data, data_size);
    return duped;
}

static void ggml_mpi_tensor_send(struct ggml_mpi_context * ctx_mpi, struct ggml_tensor * t, int mpi_rank_dst) {
    if(ctx_mpi->comm == MPI_COMM_NULL) {
        return;
    }
//    printf("Rank %d tensor send\n", ctx_mpi->rank);
    MPI_Datatype mpi_type;

    switch (t->type) {
        case GGML_TYPE_I32: mpi_type = MPI_INT32_T; break;
        case GGML_TYPE_F32: mpi_type = MPI_FLOAT;   break;
        default: GGML_ASSERT(false && "not implemented");
    }

    const int retval = MPI_Bsend(t->data, ggml_nelements(t), mpi_type, mpi_rank_dst, GGML_MPI_TRANSFER_TENSORS, ctx_mpi->comm);
    GGML_ASSERT(retval == MPI_SUCCESS);
}

static void ggml_mpi_tensor_recv(struct ggml_mpi_context * ctx_mpi, struct ggml_tensor * t, int mpi_rank_src) {
    if(ctx_mpi->comm == MPI_COMM_NULL) {
        return;
    }
    MPI_Datatype mpi_type;

    switch (t->type) {
        case GGML_TYPE_I32: mpi_type = MPI_INT32_T; break;
        case GGML_TYPE_F32: mpi_type = MPI_FLOAT;   break;
        default: GGML_ASSERT(false && "not implemented");
    }
    const int retval = MPI_Recv(t->data, ggml_nelements(t), mpi_type, mpi_rank_src, GGML_MPI_TRANSFER_TENSORS, ctx_mpi->comm, MPI_STATUS_IGNORE);
    GGML_ASSERT(retval == MPI_SUCCESS);
}

void ggml_mpi_wait_recv(struct ggml_mpi_context * ctx_mpi) {
    if(ctx_mpi->comm == MPI_COMM_NULL) {
        return;
    }
    if (ctx_mpi->asyncRecvWaiting) {
        MPI_Wait(&(ctx_mpi->asyncRecvRequest), MPI_STATUS_IGNORE);
        ctx_mpi->asyncRecvWaiting = false;
    }
}

struct ggml_tensor * ggml_mpi_async_received_tensor(struct ggml_mpi_context * ctx_mpi) {
    ggml_mpi_wait_recv(ctx_mpi);
    return ctx_mpi->duped_recv_tensor;
}

static void ggml_mpi_async_tensor_recv(struct ggml_mpi_context * ctx_mpi, struct ggml_tensor * t, int mpi_rank_src) {
    if(ctx_mpi->comm == MPI_COMM_NULL) {
        return;
    }
    MPI_Datatype mpi_type;

    switch (t->type) {
        case GGML_TYPE_I32: mpi_type = MPI_INT32_T; break;
        case GGML_TYPE_F32: mpi_type = MPI_FLOAT;   break;
        default: GGML_ASSERT(false && "not implemented");
    }

    ggml_mpi_wait_recv(ctx_mpi);
    ctx_mpi->asyncRecvWaiting = true;
    const int retval = MPI_Irecv(t->data, ggml_nelements(t), mpi_type, mpi_rank_src, GGML_MPI_TRANSFER_TENSORS, ctx_mpi->comm, &(ctx_mpi->asyncRecvRequest));

    GGML_ASSERT(retval == MPI_SUCCESS);
}

uint16_t** ggml_mpi_split_range(
    struct ggml_mpi_context * ctx_mpi,
    uint16_t start,
    uint16_t end,
    float node_weights[]
) {
    // Splits the range given by start and end
    // over the available nodes. This implementation
    // assumes that node 0 handles the final part of the range
    // while node 1 handles the beginning, to form a ring pipeline

    // Only node 0 deals with the device splits, other nodes
    // get the splits from the scatter layers operation

    if (ctx_mpi->comm == MPI_COMM_NULL || ctx_mpi->rank != 0) {
        return NULL;
    }

    uint16_t range_length = end - start + 1;
    uint16_t ** ranges = (uint16_t**) malloc(sizeof(uint16_t*) * ctx_mpi->size);
    for (int i = 0; i < ctx_mpi->size; i++) {
        ranges[i] = (uint16_t*) malloc(sizeof(uint16_t) * 2);
    }
    uint16_t next_layer = 0;
    for (int i=0; i < ctx_mpi->size-1; i++) {
        ranges[i][0] = next_layer;
        ranges[i][1] = MIN(end, ranges[i][0] + (node_weights[i] * range_length) + start);
        next_layer = ranges[i][1];
    }

    ranges[ctx_mpi->size-1][0] = next_layer;
    ranges[ctx_mpi->size-1][1] = MIN(end, next_layer + (node_weights[ctx_mpi->size-1] * range_length) + start);
    return ranges;

}

void ggml_mpi_scatter_layers(
    struct ggml_mpi_context * ctx_mpi,
    uint16_t ** layer_ranges
) {
    if(ctx_mpi->comm == MPI_COMM_NULL) {
        return;
    }

    // Layer ranges is a 2d array with the first dimension
    // having a length of the number of nodes and the second
    // dimension having a length of 2. The inner arrays contain
    // the start and end layer ID for a node.
    uint16_t flattened_ranges[ctx_mpi->size * 2];

    if (layer_ranges != NULL) {
        for (int i = 0; i < ctx_mpi->size * 2; i += 2) {
            flattened_ranges[i] = layer_ranges[i/2][0];
            flattened_ranges[i + 1] = layer_ranges[i/2][1];
        }
    }

    uint16_t received_range[2];
    MPI_Scatter(flattened_ranges, 2, MPI_UINT16_T, received_range, 2, MPI_UINT16_T, 0, ctx_mpi->comm);
    ctx_mpi->layer_start = received_range[0];
    ctx_mpi->layer_end = received_range[1];
    fprintf(stderr, "Ranges for rank %d: [%d, %d]\n", ctx_mpi->rank, ctx_mpi->layer_start, ctx_mpi->layer_end);
}


// TODO: there are many improvements that can be done to this implementation
void ggml_mpi_graph_creation_post(
        struct ggml_mpi_context * ctx_mpi,
             struct ggml_cgraph * gf,
                     const int n_layers) {
    const int mpi_rank = ctx_mpi->rank;
    const int mpi_size = ctx_mpi->size;

    struct ggml_tensor * inp_tokens = ggml_graph_get_tensor(gf, "inp_tokens");
    if (inp_tokens == NULL) {
        fprintf(stderr, "%s: tensor 'inp_tokens' not found\n", __func__);
        return;
    }

    struct ggml_tensor * inp0 = ggml_graph_get_tensor(gf, "layer_inp_0");
    if (inp0 == NULL) {
        fprintf(stderr, "%s: tensor 'inp0' not found\n", __func__);
        return;
    }

    GGML_ASSERT(inp0 == gf->nodes[0]);

//    printf("Rank %d creation post\n", mpi_rank);

    // distribute the compute graph into slices across the MPI nodes
    //
    // the main node (0) processes the last layers + the remainder of the compute graph
    // and is responsible to pass the input tokens to the first node (1)
    //
    // node 1:   [(  0) * n_per_node, (  1) * n_per_node)
    // node 2:   [(  1) * n_per_node, (  2) * n_per_node)
    // ...
    // node n-1: [(n-2) * n_per_node, (n-1) * n_per_node)
    // node 0:   [(n-1) * n_per_node,            n_nodes)
    //



    if (mpi_rank > 0) {
        // recv input data for each node into the "inp0" tensor (i.e. the first node in the compute graph)
        ggml_mpi_tensor_recv(ctx_mpi, inp0, mpi_rank - 1);

    } else if (mpi_size > 1) {
        // node 0 processes the inputs and then sends to node 1
    }

    //const int n_per_node = (n_layers + (mpi_size - 1)) / mpi_size;

    //const int il0 =               (mpi_idx + 0) * n_per_node;
    //const int il1 = MIN(n_layers, (mpi_idx + 1) * n_per_node);
    int il0 = ctx_mpi->layer_start;
    int il1 = MIN(n_layers, ctx_mpi->layer_end);

    char name_l0[GGML_MAX_NAME];
    char name_l1[GGML_MAX_NAME];

    snprintf(name_l0, sizeof(name_l0), "layer_inp_%d", il0);
    snprintf(name_l1, sizeof(name_l1), "layer_inp_%d", il1);

    const int idx_l0 =                ggml_graph_get_node_idx(gf, name_l0);
    const int idx_l1 = mpi_rank == mpi_size - 1 ? gf->n_nodes : ggml_graph_get_node_idx(gf, name_l1) + 1;

    if (idx_l0 < 0 || idx_l1 < 0) {
        fprintf(stderr, "%s: layer input nodes not found\n", __func__);
        return;
    }

    // attach the input data to all nodes that need it
    // TODO: not great - should be able to do this without modifying the compute graph (see next TODO below)
    for (int i = idx_l0; i < idx_l1; i++) {
        if (gf->nodes[i]->src[0] == gf->nodes[idx_l0]) {
            gf->nodes[i]->src[0] =  inp0;
        }
        if (gf->nodes[i]->src[1] == gf->nodes[idx_l0]) {
            gf->nodes[i]->src[1] =  inp0;
        }
    }

    // TODO: instead of rearranging the nodes, we should be able to execute a subset of the compute graph
    for (int i = 1; i < idx_l1 - idx_l0; i++) {
        gf->nodes[i] = gf->nodes[idx_l0 + i];
    }

    // the first node performs the "get_rows" operation, the rest of the nodes get the data from the previous node
    if (mpi_rank != 0 && mpi_size > 1) {
        gf->nodes[0]->op = GGML_OP_NONE;
    }

    gf->n_nodes = idx_l1 - idx_l0;


}

bool ggml_mpi_graph_compute_pre(struct ggml_mpi_context * ctx_mpi, struct ggml_cgraph * gf) {
    if (ctx_mpi->comm == MPI_COMM_NULL) {
        return false;
    }

//    printf("Rank %d compute pre\n", ctx_mpi->rank);

    const int mpi_rank = ctx_mpi->rank;
    const int mpi_size = ctx_mpi->size;

    struct ggml_tensor * inp_tokens = ggml_graph_get_tensor(gf, "inp_tokens");
    if (inp_tokens == NULL) {
        fprintf(stderr, "%s: tensor 'inp_tokens' not found\n", __func__);
        return false;
    }

    struct ggml_tensor * inp0 = ggml_graph_get_tensor(gf, "layer_inp_0");
    if (inp0 == NULL) {
        fprintf(stderr, "%s: tensor 'inp0' not found\n", __func__);
        return false;
    }

    GGML_ASSERT(inp0 == gf->nodes[0]);

    return true;
}

void ggml_mpi_graph_compute_post(
        struct ggml_mpi_context * ctx_mpi,
             struct ggml_cgraph * gf) {

    const int mpi_rank = ctx_mpi->rank;

//    printf("Rank %d compute post\n", mpi_rank);
    // send the output data to the next node
    if (mpi_rank < ctx_mpi->size - 1) {
        ggml_mpi_tensor_send(ctx_mpi, gf->nodes[gf->n_nodes - 1], ggml_mpi_next_node(ctx_mpi));
    }
}
