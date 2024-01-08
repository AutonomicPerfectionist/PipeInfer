#include "common.h"
#include "llama.h"

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>
#include <deque>
#include <stdint.h>


#define SPEC_VOCAB_MAX_SIZE_DIFFERENCE  100
#define SPEC_VOCAB_CHECK_START_TOKEN_ID 5

struct seq_draft {
    bool active   = false;
    bool drafting = false;
    bool skip     = false;

    int i_batch_dft = 0;
    std::vector<int> i_batch_tgt;

    std::vector<llama_token> tokens;
    std::vector<llama_token> prefix_tokens;

    struct llama_sampling_context * ctx_sampling;
};

struct seq_async_run {
    struct ggml_cgraph * cgraph;
    llama_batch batch;
    std::vector<seq_draft> drafts;
    int run_id;
    int n_past_tgt;
    int prefix_n_past_tgt;
    int n_past_dft;
    int i_dft;
    int s_keep;
    int seq_offset;
    int n_past_max;
    llama_sampling_context *ctx_sampling;
    bool speculative;
    bool canceled;
};

void check_for_cancel(llama_context *ctx_tgt, int n_past_tgt, std::deque<struct seq_async_run> &tgt_cgraphs,
                      std::vector<llama_token> &generated);

int main(int argc, char ** argv) {
    gpt_params params;

    if (gpt_params_parse(argc, argv, params) == false) {
        return 1;
    }

    if (params.model_draft.empty()) {
        fprintf(stderr, "%s: error: --model-draft is required\n", __func__);
        return 1;
    }

    // max number of parallel drafting sequences (i.e. tree branches)
    const int n_seq_dft = params.n_parallel;

    // probability threshold for accepting a token from the draft model
    const float p_accept = params.p_accept;

    // probability threshold for splitting a draft branch (only for n_seq_dft > 1)
    const float p_split  = params.p_split;

#ifndef LOG_DISABLE_LOGS
    log_set_target(log_filename_generator("speculative", "log"));
    LOG_TEE("Log start\n");
    log_dump_cmdline(argc, argv);
#endif // LOG_DISABLE_LOGS

    // init llama.cpp
    llama_backend_init(params.numa);

    llama_model * model_tgt = NULL;
    llama_model * model_dft = NULL;

    llama_context * ctx_tgt = NULL;
    llama_context * ctx_dft = NULL;

    // load the target model
    params.logits_all = true;
    std::tie(model_tgt, ctx_tgt) = llama_init_from_gpt_params(params);

    llama_split_comm(ctx_tgt, (llama_node_id(ctx_tgt) == 0 || llama_node_id(ctx_tgt) == params.mpi_layer_split[0].size()) ? 0 : -1);
    llama_swap_comm(ctx_tgt);

    llama_split_comm(ctx_tgt, (llama_node_id(ctx_tgt) < params.mpi_layer_split[0].size()) ? 0 : -1);
//    printf("Size of first split: %lu, element: %f\n", params.mpi_layer_split[0].size(), params.mpi_layer_split[0][0]);

    // load the draft model
    params.model = params.model_draft;
    params.n_gpu_layers = params.n_gpu_layers_draft;
    std::tie(model_dft, ctx_dft) = llama_init_from_gpt_params(params);

    llama_split_comm(ctx_dft, (llama_node_id(ctx_dft) == 0 || llama_node_id(ctx_dft) == params.mpi_layer_split[0].size()) ? 0 : -1);
    llama_swap_comm(ctx_dft);

    llama_split_comm(ctx_dft, (llama_node_id(ctx_dft) >= params.mpi_layer_split[0].size()) ? 0 : -1);

//    printf("Size of second split: %lu, element: %f\n", params.mpi_layer_split[1].size(), params.mpi_layer_split[1][0]);


    llama_split_layers_weighted(ctx_tgt, params.mpi_layer_split[0].data(), params.mpi_layer_split[0].size());
    llama_split_layers_weighted(ctx_dft, params.mpi_layer_split[1].data(), params.mpi_layer_split[1].size());

    std::deque<int> free_sequence_offsets;
    const int n_simul_seqs = 100;
    const int max_seq = n_simul_seqs * n_seq_dft + 1;
    for (int i = 0; i < n_simul_seqs; i++) {
        free_sequence_offsets.push_back(i*n_seq_dft + 1);
    }

    {
        LOG_TEE("\n");
        LOG_TEE("%s\n", get_system_info(params).c_str());
    }

    {
        const int n_vocab_tgt = llama_n_vocab(model_tgt);
        const int n_vocab_dft = llama_n_vocab(model_dft);
        const int vocab_diff  = n_vocab_tgt > n_vocab_dft
            ? n_vocab_tgt - n_vocab_dft
            : n_vocab_dft - n_vocab_tgt;

        if (vocab_diff > SPEC_VOCAB_MAX_SIZE_DIFFERENCE) {
            fprintf(stderr, "%s: error: draft model vocab must closely match target model to use speculation but ", __func__);
            fprintf(stderr, "target vocab size %d does not match draft vocab size %d - difference %d, max allowed %d\n",
                    n_vocab_tgt, llama_n_vocab(model_dft), vocab_diff, SPEC_VOCAB_MAX_SIZE_DIFFERENCE);
            return 1;
        }

//        for (int i = SPEC_VOCAB_CHECK_START_TOKEN_ID; i < std::min(n_vocab_tgt, n_vocab_dft); ++i) {
//            const char * token_text_tgt = llama_token_get_text(model_tgt, i);
//            const char * token_text_dft = llama_token_get_text(model_dft, i);
//            if (std::strcmp(token_text_tgt, token_text_dft) != 0) {
//                fprintf(stderr, "%s: error: draft model vocab must match target model to use speculation but ", __func__);
//                fprintf(stderr, "token %d content differs - target '%s', draft '%s'\n", i,
//                        llama_token_to_piece(ctx_tgt, i).c_str(),
//                        llama_token_to_piece(ctx_dft, i).c_str());
//                return 1;
//            }
//        }
    }


    // Tokenize the prompt
    const bool add_bos_tgt = llama_should_add_bos_token(model_tgt);
    LOG("add_bos tgt: %d\n", add_bos_tgt);

    const bool add_bos_dft = llama_should_add_bos_token(model_dft);
    LOG("add_bos dft: %d\n", add_bos_dft);

    if (add_bos_tgt != add_bos_dft) {
        fprintf(stderr, "%s: error: draft model add_bos must match target model to use speculation but ", __func__);
        fprintf(stderr, "add_bos_dft = %d while add_bos_tgt = %d\n", add_bos_dft, add_bos_tgt);
        return 1;
    }

    std::vector<llama_token> inp;
    inp = ::llama_tokenize(ctx_tgt, params.prompt, add_bos_tgt, true);

    const int max_context_size     = llama_n_ctx(ctx_tgt);
    const int max_tokens_list_size = max_context_size - 4;

    if ((int) inp.size() > max_tokens_list_size) {
        fprintf(stderr, "%s: error: prompt too long (%d tokens, max %d)\n", __func__, (int) inp.size(), max_tokens_list_size);
        return 1;
    }

    fprintf(stderr, "\n\n");

    if (llama_node_id(ctx_tgt) == 0) {
        for (auto id : inp) {
            fprintf(stderr, "%s", llama_token_to_piece(ctx_tgt, id).c_str());
        }
    }



    fflush(stderr);

    const int n_input = inp.size();

    const auto t_enc_start = ggml_time_us();

    int32_t batch_id = 0;

    llama_batch batch_dft = llama_batch_init(params.n_ctx, 0, max_seq);
    llama_batch batch_tgt = llama_batch_init(params.n_ctx, 0, max_seq);
    llama_batch batch_tgt_async = llama_batch_init(params.n_ctx, 0, max_seq);

    batch_dft.batch_id = batch_id;
    batch_tgt.batch_id = batch_id;
    batch_tgt_async.batch_id = batch_id;

    std::vector<llama_seq_id> seq_ids;
    for (int i = 0; i < max_seq; i++) {
        seq_ids.emplace_back(i);
    }

    for (size_t i = 0; i < inp.size()-1; i++) {
        llama_batch_add(batch_dft, inp[i], i, seq_ids, true);
        llama_batch_add(batch_tgt, inp[i], i, seq_ids, true);
    }
    llama_decode(ctx_tgt, batch_tgt);
    llama_batch_clear(batch_tgt);
    llama_batch_add(batch_dft, inp.back(), n_input-1, seq_ids, true);
    llama_batch_add(batch_tgt, inp.back(), n_input-1, seq_ids, true);

    // eval the prompt with both models
    llama_decode(ctx_tgt, batch_tgt);
    llama_decode(ctx_dft, batch_dft);

    const auto t_enc_end = ggml_time_us();

    // the 2 models should have the same vocab
    //GGML_ASSERT(n_vocab == llama_n_vocab(model_dft));

    // how many tokens to draft each time
    int n_draft = params.n_draft;

    int n_predict = 0;
    int n_drafted = 0;
    int n_accept  = 0;

    const int ASYNC_RUN_ID = n_seq_dft+1;
    int n_past_tgt = inp.size();
    int n_past_dft = inp.size();

    // used to determine end of generation
    bool has_eos = false;

    // target model sampling context
    struct llama_sampling_context * ctx_sampling = llama_sampling_init(params.sparams);

    // draft sequence data
    std::vector<seq_draft> drafts(n_seq_dft);

    params.sparams.grammar.clear(); // the draft samplers will copy the target sampler's grammar
    params.sparams.temp = -1.0f;    // force greedy sampling with probs for the draft model

    for (int s = 0; s < n_seq_dft; ++s) {
        drafts[s].ctx_sampling = llama_sampling_init(params.sparams);
    }




    std::deque<struct ggml_cgraph *> dft_cgraphs;
    std::deque<struct seq_async_run> tgt_cgraphs;

    const auto t_dec_start = ggml_time_us();

    // sample from the last token of the prompt
    drafts[0].i_batch_tgt.resize(1);
    drafts[0].i_batch_tgt[0] = 0;

    int run_id = 0;
    int offset = 1;
    int run_n_past_tgt = n_past_tgt-1;
    int run_max_n_past = n_past_tgt;
    int run_n_past_dft = n_past_dft;
    int seq_offset = free_sequence_offsets.front();
    free_sequence_offsets.pop_front();
    struct llama_kv_cache_view kvc_view = llama_kv_cache_view_init(ctx_tgt, max_seq);
    struct llama_kv_cache_view kvc_view_dft = llama_kv_cache_view_init(ctx_dft, max_seq);
    std::vector<llama_token> generated = inp;
    bool run_speculative = false;
    while (true) {


        int i_dft  = 0;
        int s_keep = 0;

        check_for_cancel(ctx_tgt, n_past_tgt, tgt_cgraphs, generated);

        if (!tgt_cgraphs.empty()) {
            struct seq_async_run run = tgt_cgraphs.back();
            LOG("Finishing async decode, is async = %d, old seq_offset = %d, new seq offset = %d, batch id = %d\n", run.run_id == ASYNC_RUN_ID, seq_offset, run.seq_offset, run.batch.batch_id);
            struct ggml_cgraph * cgraph = run.cgraph;

            run_id = run.run_id;
            drafts = run.drafts;
            run_speculative = run.speculative;
            run_max_n_past = run.n_past_max;
//            ctx_sampling = run.ctx_sampling;
            run_n_past_tgt = run.n_past_tgt;
            run_n_past_dft = run.n_past_dft;
//            n_past_dft = run.n_past_dft;
            seq_offset = run.seq_offset;


            LOG("Checking run, last generated: %d, first draft: %d\n", generated.back(), drafts[s_keep].tokens[0]);
//            if(run.n_past_max >= n_past_tgt && (!run_speculative || (n_past_tgt-run_n_past_tgt >= 0 && generated.at(generated.size() - (n_past_tgt-run_n_past_tgt+1)) == drafts[s_keep].tokens[0]))) {

            if(!run.canceled) {
                //drafts[0].tokens.erase(drafts[0].tokens.begin());
                for (int s = 0; s < n_seq_dft; ++s) {
                    if (!drafts[s].active) {
                        continue;
                    }

                    drafts[s].tokens.erase(drafts[s].tokens.begin());

                }

            } else {
//                if (run_id != ASYNC_RUN_ID) {
//                LOG("Cancelling run with ID %d, batch ID %d, run_npast_max %d, n_past_tgt %d, run_speculative %d, tokens[0] %d\n", run.run_id, run.batch.batch_id, run.n_past_max, n_past_tgt, run_speculative, drafts[s_keep].tokens[0]);
//                    llama_cancel_run(ctx_tgt, &run.batch.batch_id, 1);
//                }
                llama_finish_async_decode(*ctx_tgt, run.batch, cgraph);
                tgt_cgraphs.pop_back();
                if (run_speculative) {
                    free_sequence_offsets.push_back(seq_offset);
                }
//                fprintf(stderr, "Incorrect starting token\n");
                continue;
            }



            llama_finish_async_decode(*ctx_tgt, run.batch, cgraph);
            tgt_cgraphs.pop_back();

        }
//        if (llama_node_id(ctx_tgt) == 0) {
//            llama_kv_cache_view_update(ctx_tgt, &kvc_view);
//            dump_kv_cache_view_seqs(kvc_view, 20);
////            dump_kv_cache_view(kvc_view, 20);
//            printf("n_past_tgt: %d, run_n_past_tgt: %d, run_max_n_past: %d\n", n_past_tgt, run_n_past_tgt, run_max_n_past);
//        }
        // print current draft sequences
        bool any_active = false;
        for (int s = 0; s < n_seq_dft; ++s) {
            if (!drafts[s].active) {
                continue;
            }

            any_active = true;
            const auto & tokens = drafts[s].tokens;

            LOG("draft %d: %s\n", s, LOG_TOKENS_TOSTR_PRETTY(ctx_dft, tokens).c_str());
        }
        LOG("Any active drafts: %d\n", any_active);


        bool any_match = false;
        llama_token id;
        std::string token_str;


        int old_n_past_tgt = n_past_tgt;
        int old_n_past_dft = n_past_dft;


        std::vector<int> keeps = seq_ids;
        while (!keeps.empty()) {

            LOG("sampling target: s_keep = %3d, i_dft = %3d, i_batch_tgt = %3d, run_n_past_tgt = %3d, n_past_tgt = %3d, seq_offset = %d, keeps[0] = %d\n", s_keep, i_dft, drafts[keeps[0]].i_batch_tgt[i_dft], run_n_past_tgt, n_past_tgt, seq_offset, keeps[0]);


            // sample from the target model
            id = llama_sampling_sample(ctx_sampling, ctx_tgt, NULL, drafts[keeps[0]].i_batch_tgt[i_dft]);
            token_str = llama_token_to_piece(ctx_tgt, id);
            // Swap to pipeline roots
            llama_swap_comm(ctx_tgt);
            LOG("Swapped comm to pipeline roots, id %d\n", llama_node_id(ctx_tgt));

            llama_sync_token(ctx_tgt, &id, 0);

            LOG("Sampling index: %d\n", drafts[keeps[0]].i_batch_tgt[i_dft]);


            LOG("last: %s\n", LOG_TOKENS_TOSTR_PRETTY(ctx_tgt, ctx_sampling->prev).c_str());


            LOG("Sampled token: %d ('%s'), n_past_tgt: %d, run_n_past_tgt + i_dft: %d, drafts[keeps[0]].i_batch_tgt[i_dft]: %d\n", id, token_str.c_str(), n_past_tgt, run_n_past_tgt + i_dft, drafts[keeps[0]].i_batch_tgt[i_dft]);


            if (run_n_past_tgt + i_dft == n_past_tgt-1) {
                any_match = true;
                ++n_predict;
                llama_sampling_accept(ctx_sampling, ctx_tgt, id, true);

                // Root of WORLD
                LOG("Accepting token %d ('%s'), n_past_tgt: %d\n", id, token_str.c_str(), n_past_tgt);
                generated.push_back(id);
                if (llama_node_id(ctx_tgt) == 0) {
                    printf("%s", token_str.c_str());
                    fflush(stdout);
                }
            }

            // Switch back to target pipeline only
            llama_swap_comm(ctx_tgt);
            LOG("Swapped comm to target only, id %d\n", llama_node_id(ctx_tgt));



            if (id == llama_token_eos(model_tgt)) {
                has_eos = true;
            }





            if (run_id == ASYNC_RUN_ID) {
                break;
            }


            // check if the target token matches any of the drafts
            {
                bool matches = false;
                keeps.clear();
                for (int s = 0; s < n_seq_dft; ++s) {
                    if (!drafts[s].active) {
                        continue;
                    }

                    if (i_dft < (int) drafts[s].tokens.size() && id == drafts[s].tokens[i_dft]) {
                        LOG("the sampled target token matches the %dth drafted token of sequence %d (%d, '%s') - accepted\n", i_dft, s, id, token_str.c_str());
                        matches = true;
                        keeps.push_back(s);
                        s_keep = keeps[0];
                    } else {
                        drafts[s].active = false;
                    }
                }

                if (matches) {
                    if (run_n_past_tgt + i_dft == n_past_tgt-1) {
                        ++n_accept;
                        ++n_past_tgt;
                        ++n_past_dft;
                    }
                    ++i_dft;
                    if (run_id != ASYNC_RUN_ID && run_n_past_tgt + i_dft < n_past_tgt) {
                        continue;
                    }
                }
            }

        }



        if (llama_node_id(ctx_tgt) < 0) {
            LOG("the sampled target token (%d, '%s') did not match, or we ran out of drafted tokens\n", id, token_str.c_str());

        }

        if (!any_match) {
            if (run_id != ASYNC_RUN_ID) {
                free_sequence_offsets.push_back(seq_offset);
            }
//            fprintf(stderr, "No match\n");
            continue;
        }

        check_for_cancel(ctx_tgt, n_past_tgt, tgt_cgraphs, generated);

        // Pipeline syncing cache ops
//        llama_kv_cache_seq_keep(ctx_dft, s_keep);
//        llama_kv_cache_seq_cp  (ctx_dft, s_keep, 0, -1, -1);
//        llama_kv_cache_seq_keep(ctx_dft, 0);
//        llama_kv_cache_seq_rm  (ctx_dft, 0, n_past_dft, -1);

        // TODO: simplify
        if (run_id != ASYNC_RUN_ID){
            LOG("keeping sequence %d, n_past_tgt = %d, n_past_dft = %d\n", s_keep+seq_offset, n_past_tgt, n_past_dft);




//            llama_kv_cache_seq_rm(ctx_tgt, 0, n_past_tgt, -1);
            for (int i = 0; i < n_seq_dft; i++) {


            }
//            LOG("Copying tgt sequence %d to %d from positions %d to %d\n", s_keep+seq_offset, 0, run_n_past_tgt, n_past_tgt);
//            llama_kv_cache_seq_cp_back  (ctx_tgt, s_keep+seq_offset, 0, run_n_past_tgt, n_past_tgt);
            llama_kv_cache_seq_cp  (ctx_tgt, s_keep+seq_offset, 0, old_n_past_tgt, n_past_tgt);

//            if (llama_node_id(ctx_tgt) == 0) {
//                llama_kv_cache_view_update(ctx_tgt, &kvc_view);
//                dump_kv_cache_view_seqs(kvc_view, 20);
////            dump_kv_cache_view(kvc_view, 20);
//                printf("Copied to 0, n_past_tgt: %d, run_n_past_tgt: %d, run_max_n_past: %d\n", n_past_tgt, run_n_past_tgt, run_max_n_past);
//            }

            for (int i = 0; i < n_seq_dft; i++) {
                llama_kv_cache_seq_cp(ctx_tgt, 0, i+seq_offset, -1, n_past_tgt);
//                if (llama_node_id(ctx_tgt) == 0) {
//                    llama_kv_cache_view_update(ctx_tgt, &kvc_view);
//                    dump_kv_cache_view_seqs(kvc_view, 20);
////            dump_kv_cache_view(kvc_view, 20);
//                    printf("Copied from 0 to %d, n_past_tgt: %d, run_n_past_tgt: %d, run_max_n_past: %d\n", i+seq_offset, n_past_tgt, run_n_past_tgt, run_max_n_past);
//                }
            }

//            LOG("Copying dft sequence %d to %d from positions %d to %d\n", s_keep+seq_offset, 0, run_n_past_dft, n_past_dft);

            llama_kv_cache_seq_cp  (ctx_dft, s_keep+seq_offset, 0, old_n_past_dft, n_past_dft);
            for (int i = 0; i < n_seq_dft; i++) {
//                LOG("Removing tgt sequence %d from positions %d to %d\n", i+seq_offset, -1, -1);

                llama_kv_cache_seq_rm  (ctx_tgt, i+seq_offset, old_n_past_tgt, -1);

//                if (llama_node_id(ctx_tgt) == 0) {
//                    llama_kv_cache_view_update(ctx_tgt, &kvc_view);
//                    dump_kv_cache_view_seqs(kvc_view, 20);
////            dump_kv_cache_view(kvc_view, 20);
//                    printf("Removed %d, n_past_tgt: %d, run_n_past_tgt: %d, run_max_n_past: %d, old_n_past: %d\n", i+seq_offset, n_past_tgt, run_n_past_tgt, run_max_n_past, old_n_past_tgt);
//                }

//                LOG("Removing dft sequence %d from positions %d to %d\n", i+seq_offset, -1, -1);

                llama_kv_cache_seq_rm  (ctx_dft, i+seq_offset, old_n_past_dft, -1);
            }


            for (int i = 1; i < max_seq; i++) {
//                LOG("Copying tgt sequence %d to %d from positions %d to %d\n", 0, i, -1, n_past_tgt);
//                LOG("Copying dft sequence %d to %d from positions %d to %d\n", 0, i, -1, n_past_dft);

                llama_kv_cache_seq_rm(ctx_tgt, i, old_n_past_tgt, n_past_tgt);
                llama_kv_cache_seq_rm(ctx_dft, i, old_n_past_dft, n_past_dft);
//
//                if (llama_node_id(ctx_tgt) == 0) {
////                    llama_kv_cache_view_update(ctx_tgt, &kvc_view);
////                    dump_kv_cache_view_seqs(kvc_view, 20);
////            dump_kv_cache_view(kvc_view, 20);
//                    printf("Removed %d, n_past_tgt: %d, run_n_past_tgt: %d, run_max_n_past: %d, old_n_past: %d\n", i+seq_offset, n_past_tgt, run_n_past_tgt, run_max_n_past, old_n_past_tgt);
//                }

                llama_kv_cache_seq_cp(ctx_tgt, 0, i, old_n_past_tgt, n_past_tgt);

//                if (llama_node_id(ctx_tgt) == 0) {
////                    llama_kv_cache_view_update(ctx_tgt, &kvc_view);
////                    dump_kv_cache_view_seqs(kvc_view, 20);
////            dump_kv_cache_view(kvc_view, 20);
//                    printf("Copied 0 to %d, n_past_tgt: %d, run_n_past_tgt: %d, run_max_n_past: %d, old_n_past: %d\n", i, n_past_tgt, run_n_past_tgt, run_max_n_past, old_n_past_tgt);
//                }

                llama_kv_cache_seq_cp(ctx_dft, 0, i, old_n_past_dft, n_past_dft);
            }

//            if (llama_node_id(ctx_tgt) == 0) {
//                llama_kv_cache_view_update(ctx_tgt, &kvc_view);
//                dump_kv_cache_view_seqs(kvc_view, 20);
////            dump_kv_cache_view(kvc_view, 20);
//                printf("Kept sequence, n_past_tgt: %d, run_n_past_tgt: %d, run_max_n_past: %d\n", n_past_tgt, run_n_past_tgt, run_max_n_past);
//            }

//            for (int i = 0; i < n_seq_dft; i++) {
//                llama_kv_cache_seq_cp(ctx_tgt, 0, i+seq_offset, -1, n_past_tgt);
//            }


        } else {
//            llama_kv_cache_seq_cp  (ctx_tgt, s_keep+seq_offset, 0, -1, n_past_tgt);
//            for (int i = 1; i < n_seq_dft; i++) {
//                llama_kv_cache_seq_rm  (ctx_tgt, i+seq_offset, -1, n_past_tgt);
//
//            }

//            for (int i = 0; i < n_seq_dft; i++) {
//                llama_kv_cache_seq_rm  (ctx_tgt, i+seq_offset, -1, n_past_tgt);
//                llama_kv_cache_seq_cp(ctx_tgt, 0, i+seq_offset, -1, n_past_tgt);
//            }
        }







        {

            batch_id++;


            LOG("Beginning async decode, batch id = %d\n", batch_id);
            llama_batch_clear(batch_tgt_async);

            llama_batch_add(batch_tgt_async, id, n_past_tgt, {0}, true);

            LOG("target async batch: %s\n, batch_id = %d\n", LOG_BATCH_TOSTR_PRETTY(ctx_tgt, batch_tgt_async).c_str(), batch_id);


            // batch_tgt.n_tokens = 1

            ++n_past_tgt;
            struct seq_async_run run;
            run.canceled = false;
//            if (!free_sequence_offsets.empty()) {
//                run.seq_offset = free_sequence_offsets.front();
//                printf("Popping %d from seq offsets\n", run.seq_offset);
//                free_sequence_offsets.pop_front();
//            } else if(!tgt_cgraphs.empty()){
//                printf("Getting offset from head of tgt cgraphs\n");
//                run.seq_offset = tgt_cgraphs.front().seq_offset;
//            } else {
//                printf("NO FREE OFFSETS AND NO TGT CGRAPHS\n");
//            }
            run.batch = llama_batch_init(params.n_ctx, 0, max_seq);
            run.batch.batch_id = batch_id;
            run.batch.n_tokens = batch_tgt_async.n_tokens;
            for (int i = 0; i < batch_tgt_async.n_tokens; i++) {
                run.batch.n_seq_id[i] = batch_tgt_async.n_seq_id[i];
                for (int j = 0; j < run.batch.n_seq_id[i]; j++) {
                    run.batch.seq_id[i][j] = batch_tgt_async.seq_id[i][j];
                }
                run.batch.token[i] = batch_tgt_async.token[i];
                run.batch.pos[i] = batch_tgt_async.pos[i];
                run.batch.logits[i] = batch_tgt_async.logits[i];
            }
            run.ctx_sampling = llama_sampling_init(params.sparams);
            llama_sampling_cp(ctx_sampling, run.ctx_sampling);
            run.drafts = std::vector<seq_draft>(n_seq_dft);
            for (int s = 0; s < n_seq_dft; ++s) {
                run.drafts[s].ctx_sampling = llama_sampling_init(params.sparams);
                llama_sampling_cp(drafts[s].ctx_sampling, run.drafts[s].ctx_sampling);
                run.drafts[s].i_batch_tgt = std::vector<int>(1,0);
                run.drafts[s].i_batch_dft = drafts[s].i_batch_dft;
                run.drafts[s].tokens = std::vector<llama_token>(1, id);
                run.drafts[s].active = drafts[s].active;
                run.drafts[s].drafting = drafts[s].drafting;
                run.drafts[s].skip = drafts[s].skip;
                run.drafts[s].prefix_tokens = std::vector<llama_token>(0);
            }
            run.i_dft = offset - 1;
            run.s_keep = s_keep;
            run.run_id = ASYNC_RUN_ID;
            run.n_past_tgt = n_past_tgt-1;
            run.prefix_n_past_tgt = n_past_tgt-1;
            run.n_past_max = n_past_tgt;
            run.n_past_dft = n_past_dft;
            run.speculative = false;
            run.cgraph = llama_start_async_decode(*ctx_tgt, run.batch);
            tgt_cgraphs.push_front(run);
            //llama_kv_cache_seq_rm(ctx_tgt, 0, n_past_tgt, n_past_tgt+1);
            for (int i = 1; i < max_seq; i++) {
//                LOG("Copying tgt sequence %d to %d from positions %d to %d\n", 0, i, n_past_tgt-1, n_past_tgt);
                llama_kv_cache_seq_rm(ctx_tgt, i, n_past_tgt-1, n_past_tgt);
                llama_kv_cache_seq_cp(ctx_tgt, 0, i, n_past_tgt-1, n_past_tgt);
            }

//            if (llama_node_id(ctx_tgt) == 0) {
//                llama_kv_cache_view_update(ctx_tgt, &kvc_view);
//                dump_kv_cache_view_seqs(kvc_view, 20);
////            dump_kv_cache_view(kvc_view, 20);
//                printf("Copied async, n_past_tgt: %d, run_n_past_tgt: %d, run_max_n_past: %d\n", n_past_tgt, run_n_past_tgt, run_max_n_past);
//            }
        }


        if (run_id == ASYNC_RUN_ID) {
            if (free_sequence_offsets.empty()) {
                continue;
            }
            seq_offset = free_sequence_offsets.front();
//            printf("Popping %d from seq offsets for spec run\n", seq_offset);
            free_sequence_offsets.pop_front();
        }

        int spec_past_tgt = n_past_tgt;
        int spec_past_dft = n_past_dft;

        int first_run = true;

//        bool is_waiting = llama_mpi_iprobe(ctx_tgt);
        bool is_waiting = false;
//        llama_swap_comm(ctx_tgt);
//        llama_sync_token(ctx_tgt, reinterpret_cast<llama_token *>(&is_waiting), 0);
//        llama_swap_comm(ctx_tgt);


//        llama_batch_clear(batch_dft);
//        llama_batch_add(batch_dft, id, spec_past_dft, {0}, true);
//        // batch_dft.n_tokens == 1 now
//
//
//
//        // Kick off drafting pipeline but don't need it just yet
//        LOG("Beginning async draft with sequence 0\n");
//        dft_cgraphs.push_front(llama_start_async_decode(*ctx_dft, batch_dft));
//        // DON'T FORGET THE MATCHING DECODE WHEN NEEDED
//
//        // We need the draft now, so wait for it
//        if (!dft_cgraphs.empty()) {
////            LOG("Finishing async decode of draft\n");
//            llama_finish_async_decode(*ctx_dft, batch_dft, dft_cgraphs.back());
//            dft_cgraphs.pop_back();
//        }
//        LOG("dft batch: %s\n", LOG_BATCH_TOSTR_PRETTY(ctx_dft, batch_dft).c_str());


        if (is_waiting) {
            free_sequence_offsets.push_back(seq_offset);
        }
        int iter = 0;
        while(iter < 2 || !is_waiting) {





            int orig_offset = seq_offset;
            bool should_run_spec = true;
            std::deque<int> checked_offsets;
            do {
                should_run_spec = true;
                for (const auto &r: tgt_cgraphs) {
                    if (r.seq_offset == seq_offset && r.run_id != ASYNC_RUN_ID) {
                        checked_offsets.push_back(seq_offset);

                        should_run_spec = false;
                        if (!free_sequence_offsets.empty()) {
                            seq_offset = free_sequence_offsets.front();
                            free_sequence_offsets.pop_front();

                        }
                        break;
                    }
                }
            } while (!should_run_spec && !free_sequence_offsets.empty());

            free_sequence_offsets.insert(free_sequence_offsets.end(), checked_offsets.begin(),
                                         checked_offsets.end());

            if (!should_run_spec) {

                break;
            }

            LOG("Doing speculative run, seq_offset = %d, spec_past_tgt = %d, spec_past_dft = %d, n_past_tgt = %d, n_past_dft = %d", seq_offset, spec_past_tgt, spec_past_dft, n_past_tgt, n_past_dft);

            for (int i = 0; i < n_seq_dft; i++) {
//                LOG("Removing tgt sequence %d from positions %d to %d\n", i + seq_offset, -1, -1);

//                if(first_run) {

                    llama_kv_cache_seq_rm(ctx_tgt, i + seq_offset, -1, -1);

                    LOG("Copying tgt sequence %d to %d from positions %d to %d\n", (first_run) ? 0 : orig_offset,
                        i + seq_offset, -1, (first_run) ? spec_past_tgt : spec_past_tgt);

                    llama_kv_cache_seq_cp(ctx_tgt, (first_run) ? 0 : orig_offset, i + seq_offset, -1, (first_run) ? spec_past_tgt : spec_past_tgt);
//                if (llama_node_id(ctx_tgt) == 0) {
//                    llama_kv_cache_view_update(ctx_tgt, &kvc_view);
//                    dump_kv_cache_view_seqs(kvc_view, 20);
////            dump_kv_cache_view(kvc_view, 20);
//                    printf("Copied tgt sequence, n_past_tgt: %d, run_n_past_tgt: %d, run_max_n_past: %d, spec_past_tgt: %d, first_run: %d\n", n_past_tgt, run_n_past_tgt, run_max_n_past, spec_past_tgt, first_run);
//                }
//                }

//                LOG("Removing dft sequence %d from positions %d to %d\n", i + seq_offset, spec_past_dft, -1);


                    llama_kv_cache_seq_rm(ctx_dft, i + seq_offset, spec_past_dft, -1);

            }


            llama_batch_clear(batch_tgt);

            for (int s = 0; s < n_seq_dft; ++s) {
                drafts[s].active = false;
                if (!first_run) {
                    if (!drafts[s].tokens.empty()) {
                        drafts[s].prefix_tokens.insert(drafts[s].prefix_tokens.end(), drafts[s].tokens.begin(),
                                                       drafts[s].tokens.end());
                    }
                } else {
                    drafts[s].prefix_tokens.clear();
                }
                drafts[s].tokens.clear();
                drafts[s].i_batch_tgt.clear();
            }
            // note: will be erased after the speculation phase
            drafts[0].tokens.push_back(id);


//            LOG("Copying dft sequence %d to %d from positions %d to %d\n", (first_run) ? 0 : orig_offset, seq_offset, -1, spec_past_dft);

            llama_kv_cache_seq_cp(ctx_dft, (first_run) ? 0 : orig_offset, seq_offset, -1, (first_run) ? spec_past_dft : spec_past_dft + 1);

            llama_batch_clear(batch_dft);


            if (first_run) {
                llama_batch_add(batch_dft, id, spec_past_dft, {seq_offset}, true);
                // batch_dft.n_tokens == 1 now



                // Kick off drafting pipeline but don't need it just yet
                LOG("Beginning async draft\n");
                dft_cgraphs.push_front(llama_start_async_decode(*ctx_dft, batch_dft));
                // DON'T FORGET THE MATCHING DECODE WHEN NEEDED

                // We need the draft now, so wait for it
                if (!dft_cgraphs.empty()) {
                    LOG("Finishing async decode of draft\n");
                    llama_finish_async_decode(*ctx_dft, batch_dft, dft_cgraphs.back());
                    dft_cgraphs.pop_back();
                }
                LOG("dft batch: %s\n", LOG_BATCH_TOSTR_PRETTY(ctx_dft, batch_dft).c_str());

                for (int i = 0; i < max_seq; i++) {
//                    LOG("Copying dft sequence %d to %d from positions %d to %d\n", seq_offset, i, spec_past_dft, spec_past_dft + 1);

                    if (i != seq_offset) {
                        llama_kv_cache_seq_cp(ctx_dft, seq_offset, i, spec_past_dft, spec_past_dft + 1);
                    }
                }
                ++n_past_dft;
            } else {
                for (int i = 1; i < n_seq_dft; i++) {
//                    LOG("Copying dft sequence %d to %d from positions %d to %d\n", seq_offset, i+seq_offset, spec_past_dft, spec_past_dft + 1);

                    llama_kv_cache_seq_cp(ctx_dft, seq_offset, i+seq_offset, -1, spec_past_dft + 1);
                }
            }

            if (llama_node_id(ctx_dft) == 0) {
//            llama_kv_cache_view_update(ctx_dft, &kvc_view_dft);
//            dump_kv_cache_view_seqs(kvc_view_dft, 20);
            }



            if (n_predict > params.n_predict || has_eos) {
                break;
            }

            llama_sampling_cp(ctx_sampling, drafts[0].ctx_sampling);

            int n_seq_cur  = 0;
            int max_ran_seq = 0;
            int n_past_cur = spec_past_dft+1;

            for (int s = 0; s < n_seq_dft; ++s) {
                drafts[s].skip = true;
                drafts[s].active = false;
                drafts[s].drafting = false;
            }


            drafts[0].active = true;
            drafts[0].drafting = true;
            drafts[0].skip = false;

            drafts[0].i_batch_dft = 0;


            // sample n_draft tokens from the draft model using tree-based sampling
            for (int i = 0; i < n_draft; ++i) {
                batch_dft.n_tokens = 0;

                for (int s = 0; s <= max_ran_seq; ++s) {
                    if (!drafts[s].drafting || drafts[s].skip) {
                        continue;
                    }



                    // Swap back to pipeline roots
                    llama_swap_comm(ctx_dft);
                    LOG("Swapped comm to pipeline roots, id %d\n", llama_node_id(ctx_dft));

                    llama_sync_token(ctx_dft, &(drafts[s].i_batch_dft), 1);

                    llama_sampling_sample(drafts[s].ctx_sampling, ctx_dft, NULL, drafts[s].i_batch_dft);

                    auto &cur_p = drafts[s].ctx_sampling->cur;

                    llama_sync_token_data(ctx_dft, cur_p.data(), 1);
                    // TODO investigate potential bottleneck
                    for (int k = 1; k < 8; ++k) {
                        llama_sync_token_data(ctx_dft, &(cur_p[k]), 1);
                    }

                    // Back to draft pipeline only
                    llama_swap_comm(ctx_dft);
                    LOG("Swapped comm to draft only, id %d\n", llama_node_id(ctx_dft));


                    if (llama_node_id(ctx_dft) >= 0) {
                        for (int k = 0; k < std::min(n_seq_dft + 3, (int) cur_p.size()); ++k) {
                            LOG(" - draft candidate %3d for seq %3d, pos %3d: %6d (%8.3f) '%s'\n",
                                k, s, i, cur_p[k].id, cur_p[k].p, llama_token_to_piece(ctx_dft, cur_p[k].id).c_str());
                        }
                    }


                    if (cur_p[0].p < p_accept) {
                        LOG("stopping drafting for seq %3d, probability too low: %.3f < %.3f\n", s, cur_p[0].p,
                            p_accept);
                        drafts[s].drafting = false;
                        continue;
                    }


                    std::vector<int> sa(1, s);

                    // attempt to split the branch if the probability is high enough
                    for (int f = 1; f < 8; ++f) {
                        if (n_seq_cur < n_seq_dft - 1 && cur_p[f].p > p_split) {
                            n_seq_cur++;
                            LOG("splitting seq %3d into %3d\n", s, n_seq_cur);


//                            LOG("Removing dft sequence %d from positions %d to %d\n", n_seq_cur + seq_offset, -1, -1);

                            llama_kv_cache_seq_rm(ctx_dft, n_seq_cur + seq_offset, -1, -1);

//                            LOG("Copying dft sequence %d to %d from positions %d to %d\n", s + seq_offset, n_seq_cur + seq_offset, -1, -1);

                            llama_kv_cache_seq_cp(ctx_dft, s + seq_offset, n_seq_cur + seq_offset, -1, -1);

                            // all previous tokens from this branch are now also part of the new branch
                            for (int t = 0; t < batch_tgt.n_tokens; ++t) {
                                for (int p = 0; p < batch_tgt.n_seq_id[t]; ++p) {
                                    if (batch_tgt.seq_id[t][p] == s + seq_offset) {
                                        batch_tgt.seq_id[t][batch_tgt.n_seq_id[t]] = n_seq_cur + seq_offset;
                                        batch_tgt.n_seq_id[t]++;
                                        break;
                                    }
                                }
                            }


                            // copy the draft state
                            drafts[n_seq_cur].active = true;
                            drafts[n_seq_cur].drafting = true;
                            drafts[n_seq_cur].skip = false;

                            drafts[n_seq_cur].tokens = drafts[s].tokens;
                            drafts[n_seq_cur].i_batch_tgt = drafts[s].i_batch_tgt;

                            llama_sampling_cp(drafts[s].ctx_sampling, drafts[n_seq_cur].ctx_sampling);

                            sa.push_back(n_seq_cur);


                        } else {
                            break;
                        }
                    }

                    // add drafted token for each sequence
                    // TODO commenting this out fixes async
                    for (int is = 0; is < (int) sa.size(); ++is) {
                        const llama_token id = cur_p[is].id;

                        const int s = sa[is];

                        llama_sampling_accept(drafts[s].ctx_sampling, ctx_dft, id, true);

                        drafts[s].tokens.push_back(id);

                        // add unique drafted tokens to the target batch

                        drafts[s].i_batch_tgt.push_back(batch_tgt.n_tokens);

                        LOG("Adding drafted token %d to tgt, sequence %d, position %d, i_batch_tgt %d\n", id,
                            s + seq_offset, spec_past_tgt + i, batch_tgt.n_tokens);
                        llama_batch_add(batch_tgt, id, spec_past_tgt + i, {s + seq_offset}, true);

                        // add the token to the batch for batched decoding with the draft model
                        drafts[s].i_batch_dft = batch_dft.n_tokens;

                        LOG("Adding drafted token %d to dft\n", id);

                        llama_batch_add(batch_dft, id, n_past_cur, {s + seq_offset}, true);

                        if (batch_tgt.n_tokens > n_draft) {
                            drafts[s].drafting = false;
                        }
                    }
                }

                // no sequence is drafting anymore
                if (batch_dft.n_tokens == 0) {
                    break;
                }

                // evaluate the drafted tokens on the draft model
                LOG("Running synchronous draft decode\n");
                llama_decode(ctx_dft, batch_dft);
                ++n_past_cur;
                ++n_drafted;

                max_ran_seq = n_seq_cur;

                llama_batch_clear(batch_dft);

                if (batch_tgt.n_tokens > n_draft) {
                    break;
                }
            }

            // no sequence is drafting anymore
            if (batch_dft.n_tokens != 0) {
                // evaluate the drafted tokens on the draft model
                LOG("Running synchronous draft decode\n");
                llama_decode(ctx_dft, batch_dft);

            }







            // evaluate the target model on the drafted tokens
            {
//            llama_kv_cache_seq_keep(ctx_tgt, 0); // Needed to get to "Here's the code:"





                if (batch_tgt.n_tokens == 0) {
                    free_sequence_offsets.push_back(seq_offset);
                    break;
                }

                size_t max_draft_tokens = 0;

                for (int s = 0; s < n_seq_dft; ++s) {
                    if (!drafts[s].active) {
                        continue;
                    }

                    drafts[s].tokens.erase(drafts[s].tokens.begin());
                    max_draft_tokens = std::max(max_draft_tokens, drafts[s].tokens.size());
                    //drafts[s].tokens.erase(drafts[s].tokens.begin());
                }

                batch_id++;


                LOG("target batch: %s\n, batch_id = %d\n", LOG_BATCH_TOSTR_PRETTY(ctx_tgt, batch_tgt).c_str(), batch_id);
                struct seq_async_run run;
                run.canceled = false;
                run.speculative = true;
                run.seq_offset = seq_offset;
                run.ctx_sampling = llama_sampling_init(params.sparams);
                llama_sampling_cp(ctx_sampling, run.ctx_sampling);
                run.drafts = std::vector<seq_draft>(n_seq_dft);
                for (int s = 0; s < n_seq_dft; ++s) {
                    run.drafts[s].ctx_sampling = llama_sampling_init(params.sparams);
                    llama_sampling_cp(drafts[s].ctx_sampling, run.drafts[s].ctx_sampling);
                    run.drafts[s].i_batch_tgt = drafts[s].i_batch_tgt;
                    run.drafts[s].tokens = drafts[s].tokens;
                    run.drafts[s].active = drafts[s].active;
                    run.drafts[s].drafting = drafts[s].drafting;
                    run.drafts[s].skip = drafts[s].skip;
                    run.drafts[s].i_batch_dft = drafts[s].i_batch_dft;
                    run.drafts[s].prefix_tokens = drafts[s].prefix_tokens;
                }
                run.i_dft = offset;
                run.s_keep = s_keep;
                run.batch = llama_batch_init(params.n_ctx, 0, max_seq);
                run.batch.batch_id = batch_id;
                run.batch.n_tokens = batch_tgt.n_tokens;
                for (int i = 0; i < batch_tgt.n_tokens; i++) {
                    run.batch.n_seq_id[i] = batch_tgt.n_seq_id[i];
                    int cur_n_seqs = 0;
                    for (int j = 0; j < run.batch.n_seq_id[i]; j++) {
                        run.batch.seq_id[i][j] = batch_tgt.seq_id[i][j];
                    }
                    run.batch.token[i] = batch_tgt.token[i];
                    run.batch.pos[i] = batch_tgt.pos[i];
                    run.batch.logits[i] = batch_tgt.logits[i];
                }
                run.run_id = 0;
                run.n_past_tgt = spec_past_tgt;
                run.prefix_n_past_tgt = n_past_tgt;
                run.n_past_dft = n_past_dft;
                run.n_past_max = spec_past_tgt + max_draft_tokens;
                run.cgraph = llama_start_async_decode(*ctx_tgt, run.batch);
                tgt_cgraphs.push_front(run);

                spec_past_tgt += run.drafts[0].tokens.size();
                spec_past_dft += run.drafts[0].tokens.size();
                id = run.drafts[0].tokens.back();
                first_run = false;

            }

            is_waiting = llama_mpi_iprobe(ctx_tgt);
            llama_swap_comm(ctx_tgt);
            llama_sync_token(ctx_tgt, reinterpret_cast<llama_token *>(&is_waiting), 0);
            llama_swap_comm(ctx_tgt);

            iter++;
//            break;

        }

        if (n_predict > params.n_predict || has_eos) {
            break;
        }




    }

    auto t_dec_end = ggml_time_us();

    LOG_TEE("\n\n");

    LOG_TEE("encoded %4d tokens in %8.3f seconds, speed: %8.3f t/s\n", n_input,   (t_enc_end - t_enc_start) / 1e6f, inp.size() / ((t_enc_end - t_enc_start) / 1e6f));
    LOG_TEE("decoded %4d tokens in %8.3f seconds, speed: %8.3f t/s\n", n_predict, (t_dec_end - t_dec_start) / 1e6f, n_predict  / ((t_dec_end - t_dec_start) / 1e6f));

    LOG_TEE("\n");
    LOG_TEE("n_draft   = %d\n", n_draft);
    LOG_TEE("n_predict = %d\n", n_predict);
    LOG_TEE("n_drafted = %d\n", n_drafted);
    LOG_TEE("n_accept  = %d\n", n_accept);
    LOG_TEE("accept    = %.3f%%\n", 100.0f * n_accept / n_drafted);

    LOG_TEE("\ndraft:\n");
    llama_print_timings(ctx_dft);

    LOG_TEE("\ntarget:\n");
    llama_print_timings(ctx_tgt);

    llama_sampling_free(ctx_sampling);
    for (int s = 0; s < n_seq_dft; ++s) {
        llama_sampling_free(drafts[s].ctx_sampling);
    }

    llama_batch_free(batch_dft);

    llama_free(ctx_tgt);
    llama_free_model(model_tgt);

    llama_free(ctx_dft);
    llama_free_model(model_dft);

    llama_backend_free();

    fprintf(stderr, "\n\n");

    return 0;
}

void check_for_cancel(llama_context *ctx_tgt, int n_past_tgt, std::deque<struct seq_async_run> &tgt_cgraphs,
                      std::vector<llama_token> &generated) {
    std::vector<int> canceled_batches;
    for (auto &run : tgt_cgraphs) {
        if(!run.canceled) {
            bool correct_prefix = true;

            if (run.speculative && n_past_tgt >= run.prefix_n_past_tgt) {
                size_t draft_index = 0;
                int prev_token = -1;
                int prev_gen_token = -1;
                std::vector<llama_token> concat_tokens = run.drafts[0].prefix_tokens;
                concat_tokens.insert(concat_tokens.end(), run.drafts[0].tokens.begin(),
                                     run.drafts[0].tokens.end());


                LOG("Prefix tokens: %s\n", LOG_TOKENS_TOSTR_PRETTY(ctx_tgt, run.drafts[0].prefix_tokens).c_str());

                LOG("Concat tokens: %s\n", LOG_TOKENS_TOSTR_PRETTY(ctx_tgt, concat_tokens).c_str());


                size_t index = run.prefix_n_past_tgt + draft_index;
                LOG("Looping over run starting at gen index %zu, draft index %zu, prefix_n_past_tgt %d, n_past_tgt %d, generated size %zu\n", index, draft_index, run.prefix_n_past_tgt, n_past_tgt, generated.size());
                while (index < generated.size() && draft_index < concat_tokens.size() && generated.size() > (size_t)run.prefix_n_past_tgt) {
                    LOG("Checking draft at index %zu and generated index %zu\n", draft_index, index);
                    if (generated.at(index) != concat_tokens[draft_index]) {
                        LOG("Found non-matching prefix at generated index %zu, draft index %zu, gen token %d, draft token %d, prev draft token %d, prev gen token %d\n", index, draft_index, generated.at(index), concat_tokens[draft_index], prev_token, prev_gen_token);
                        correct_prefix = false;
                        break;
                    }
                    prev_token = concat_tokens[draft_index];
                    prev_gen_token = generated[index];
                    draft_index++;
                    index = run.prefix_n_past_tgt + draft_index;
                }
            }


            if (run.n_past_max < n_past_tgt || !correct_prefix) {
                LOG("Cancelling run with ID %d, batch ID %d, run.npast_max %d, run.n_past_tgt %d, n_past_tgt %d, run_speculative %d, tokens[0] %d, generated: %d, generated index: %zu\n",
                    run.run_id, run.batch.batch_id, run.n_past_max, run.n_past_tgt, n_past_tgt, run.speculative,
                    run.drafts[0].tokens[0], (n_past_tgt < run.n_past_tgt) ? -1 : generated.at(
                        generated.size() - (n_past_tgt - run.n_past_tgt + 1)),
                    generated.size() - (n_past_tgt - run.n_past_tgt + 1));

                if (run.speculative) {
                    // TODO put these in a vector so they are transmitted in a burst
                    canceled_batches.push_back(run.batch.batch_id);
                }
                run.canceled = true;
////                }
//
//                if (run_speculative) {
//                    free_sequence_offsets.push_back(seq_offset);
//                }
            }
        }
    }

    if (!canceled_batches.empty()) {
        llama_cancel_run(ctx_tgt, canceled_batches.data(), canceled_batches.size());
    }
}
