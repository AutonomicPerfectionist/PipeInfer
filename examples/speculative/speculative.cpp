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
    int n_past_tgt;
    int prefix_n_past_tgt;
    int n_past_dft;
    int i_dft;
    int s_keep;
    int seq_offset;
    int n_past_max;
    bool speculative;
    bool canceled;
};


void check_for_cancel(llama_context *ctx_tgt, int n_past_tgt, std::deque<struct seq_async_run> &tgt_cgraphs,
                      std::vector<llama_token> &generated, int n_seq_dft);

void begin_async_run(const llama_sampling_params& sparams, int n_seq_dft,
                     llama_context *ctx_tgt, int max_seq,
                     int n_past_dft, const std::vector<seq_draft> &drafts,
                     std::deque<struct seq_async_run> &tgt_cgraphs,
                     int32_t &batch_id, int &n_past, llama_kv_cache_view &kvc_view,
                     bool is_spec, llama_batch batch, int n_past_max, int prefix_n_past, int seq_offset);

bool start_async_spec_run(const gpt_params &params, llama_context *ctx_tgt, llama_context *ctx_dft,
                          std::deque<int> &free_sequence_offsets, int max_seq, llama_batch &batch_tgt, int n_predict,
                          int prefix_n_past, int n_past_dft, bool has_eos, llama_sampling_context *ctx_sampling,
                          std::deque<struct seq_async_run> &tgt_cgraphs, const seq_async_run &current_run,
                          int &spec_past_tgt, int &spec_past_dft, int first_run, int orig_offset, int32_t &batch_id,
                          llama_batch &batch_dft, int &n_drafted, std::vector<seq_draft> &drafts, llama_token &id,
                          llama_kv_cache_view &kvc, int iter);

void begin_non_spec_run(const gpt_params &params, int n_seq_dft, llama_context *ctx, int max_seq,
                        const std::vector<seq_draft> &drafts, llama_token id, int32_t &batch_id, int &n_past, int n_past_dft,
                        std::deque<struct seq_async_run> &dft_cgraphs, llama_kv_cache_view &kvc_view);

void
run_speculation_loop(const gpt_params &params, const float p_accept, llama_context *ctx_tgt, llama_context *ctx_dft,
                     int max_seq, llama_batch &batch_tgt, int n_predict, int n_past_tgt, int n_past_dft,
                     bool has_eos, llama_sampling_context *ctx_sampling, int & spec_past_tgt, int & spec_past_dft,
                     bool & first_run, std::deque<int> &free_sequence_offsets, int32_t &batch_id, llama_batch &batch_dft,
                     int &n_drafted, std::vector<seq_draft> &drafts, std::deque<struct seq_async_run> &tgt_cgraphs,
                     seq_async_run &current_run, llama_kv_cache_view &kvc_view_dft, llama_token &id);

int main(int argc, char ** argv) {
    gpt_params params;

    if (!gpt_params_parse(argc, argv, params)) {
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
    const int n_simul_seqs = 1000;
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

    llama_batch batch_dft = llama_batch_init(params.n_ctx, 0, max_seq+1);
    llama_batch batch_tgt = llama_batch_init(params.n_ctx, 0, max_seq+1);
    llama_batch batch_tgt_async = llama_batch_init(params.n_ctx, 0, max_seq+1);

    batch_dft.batch_id = batch_id;
    batch_tgt.batch_id = batch_id;
    batch_tgt_async.batch_id = batch_id;

    std::vector<llama_seq_id> seq_ids;
    for (int i = 0; i <= max_seq; i++) {
        seq_ids.emplace_back(i);
    }

    for (int i = 0; i < n_input-1; i++) {
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


    int n_past_tgt = n_input;
    int n_past_dft = n_input;

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




    std::deque<struct seq_async_run> dft_cgraphs;
    std::deque<struct seq_async_run> tgt_cgraphs;

    const auto t_dec_start = ggml_time_us();

    // sample from the last token of the prompt
    drafts[0].i_batch_tgt.resize(1);
    drafts[0].i_batch_tgt[0] = 0;

    seq_async_run current_run;

    current_run.n_past_tgt = n_past_tgt - 1;
    current_run.n_past_max = n_past_tgt;
    current_run.n_past_dft = n_past_dft - 1;
    current_run.seq_offset = free_sequence_offsets.front();
    struct llama_kv_cache_view kvc_view = llama_kv_cache_view_init(ctx_tgt, max_seq+1);
    struct llama_kv_cache_view kvc_view_dft = llama_kv_cache_view_init(ctx_dft, max_seq+1);
    std::vector<llama_token> generated = inp;

    int spec_past_tgt = n_past_tgt;
    int spec_past_dft = n_past_dft;

    long ttft = ggml_time_us();
    std::vector<uint64_t > inter_token_times;
    int64_t itt_start;
    bool first_token = false;
    bool has_run_first_token = false;

    bool first_run = true;
    llama_token id;
    while (true) {


        int i_dft  = 0;
        int s_keep = 0;


        bool is_waiting = llama_mpi_iprobe(ctx_tgt);
        llama_swap_comm(ctx_tgt);
        llama_sync_token(ctx_tgt, reinterpret_cast<llama_token *>(&is_waiting), 0);
        llama_swap_comm(ctx_tgt);

        if (!tgt_cgraphs.empty() && is_waiting) {
            check_for_cancel(ctx_tgt, n_past_tgt, tgt_cgraphs, generated, n_seq_dft);

            struct seq_async_run run = tgt_cgraphs.back();
            LOG("Finishing async decode, is spec = %d, old seq_offset = %d, new seq offset = %d, batch id = %d\n", run.speculative, current_run.seq_offset, run.seq_offset, run.batch.batch_id);
            struct ggml_cgraph * cgraph = run.cgraph;



            LOG("Checking run, last generated: %d, first draft: %d\n", generated.back(), run.drafts[run.s_keep].tokens[0]);
//            if(run.n_past_max >= n_past_tgt && (!run_speculative || (n_past_tgt-current_run.n_past_tgt >= 0 && generated.at(generated.size() - (n_past_tgt-current_run.n_past_tgt+1)) == drafts[s_keep].tokens[0]))) {

            if(!run.canceled) {

                drafts = run.drafts;
                current_run.speculative = run.speculative;
                current_run.n_past_max = run.n_past_max;
                current_run.n_past_tgt = run.n_past_tgt;
                current_run.n_past_dft = run.n_past_dft;
                current_run.seq_offset = run.seq_offset;
                s_keep = run.s_keep;

                //drafts[0].tokens.erase(drafts[0].tokens.begin());
                for (int s = 0; s < n_seq_dft; ++s) {
                    if (!drafts[s].active) {
                        continue;
                    }

                    drafts[s].tokens.erase(drafts[s].tokens.begin());

                }

            } else {
//                if (llama_node_id(ctx_tgt) == 0) {
//                    printf("\nFinishing canceled async run, spec: %d, batch id: %d, batch: %s\n", run.speculative, run.batch.batch_id, LOG_BATCH_TOSTR_PRETTY(ctx_tgt, run.batch).c_str());
//                }
// FIXME Main bottleneck because when finishing a canceled run, we're forced to wait until a correct run
//  is finished instead of jumping back to speculation
                llama_finish_async_decode(*ctx_tgt, run.batch, cgraph);
                tgt_cgraphs.pop_back();
                if (run.speculative) {
//                    if(llama_node_id(ctx_tgt) == 0) {
//                        fprintf(stderr, "\nRun was canceled, pushing seq offset %d to free seq offsets\n",
//                                run.seq_offset);
//                        fflush(stderr);
//                    }
                    free_sequence_offsets.push_back(run.seq_offset);
//                    if(llama_node_id(ctx_tgt) == 0) {
//
//                        fprintf(stderr, "\nDone pushing seq offset %d to free seq offsets\n", run.seq_offset);
//                        fflush(stderr);
//                    }
                }
//                fprintf(stderr, "Incorrect starting token\n");
                continue;
            }


//            if (llama_node_id(ctx_tgt) == 0) {
//                printf("\nFinishing async run, spec: %d, batch id: %d, batch: %s\n", run.speculative, run.batch.batch_id, LOG_BATCH_TOSTR_PRETTY(ctx_tgt, run.batch).c_str());
//            }
            llama_finish_async_decode(*ctx_tgt, run.batch, cgraph);
            tgt_cgraphs.pop_back();

            spec_past_tgt = n_past_tgt;
            spec_past_dft = n_past_dft;

            first_run = true;

        } else if (!tgt_cgraphs.empty()) {
            run_speculation_loop(params, p_accept, ctx_tgt, ctx_dft, max_seq, batch_tgt, n_predict, n_past_tgt, n_past_dft,
                                 has_eos, ctx_sampling,
                                 spec_past_tgt, spec_past_dft, first_run, free_sequence_offsets, batch_id, batch_dft,
                                 n_drafted, drafts, tgt_cgraphs, current_run, kvc_view_dft, id);
            continue;
        }


        if (llama_node_id(ctx_tgt) == 0) {
//            llama_kv_cache_view_update(ctx_tgt, &kvc_view);
//            LOG("Beginning sampling, tgt cache layout:\n%s", dump_kv_cache_view_seqs(kvc_view, 1).c_str());
            LOG("n_past_tgt: %d, current_run.n_past_tgt: %d, current_run.n_past_max: %d\n", n_past_tgt, current_run.n_past_tgt, current_run.n_past_max);
        } else {
//            llama_kv_cache_view_update(ctx_dft, &kvc_view_dft);
//            LOG("Beginning sampling, dft cache layout:\n%s", dump_kv_cache_view_seqs(kvc_view_dft, 1).c_str());
            LOG("n_past_dft: %d, current_run.n_past_dft: %d, current_run.n_past_max: %d\n", n_past_dft, current_run.n_past_dft, current_run.n_past_max);
        }
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

        std::string token_str;


        int old_n_past_tgt = n_past_tgt;
        int old_n_past_dft = n_past_dft;


        std::deque<int> keeps(seq_ids.begin(), seq_ids.end());
        keeps.erase(std::find(keeps.begin(), keeps.end(),s_keep));
        keeps.push_front(s_keep);
        while (!keeps.empty()) {

            LOG("sampling target: s_keep = %3d, i_dft = %3d, i_batch_tgt = %3d, current_run.n_past_tgt = %3d, n_past_tgt = %3d, seq_offset = %d, keeps[0] = %d\n", s_keep, i_dft, drafts[keeps[0]].i_batch_tgt[i_dft], current_run.n_past_tgt, n_past_tgt, current_run.seq_offset, keeps[0]);


            // sample from the target model
            id = llama_sampling_sample(ctx_sampling, ctx_tgt, nullptr, drafts[keeps[0]].i_batch_tgt[i_dft]);
            token_str = llama_token_to_piece(ctx_tgt, id);
            // Swap to pipeline roots
            llama_swap_comm(ctx_tgt);
            LOG("Swapped comm to pipeline roots, id %d\n", llama_node_id(ctx_tgt));

            llama_sync_token(ctx_tgt, &id, 0);

            LOG("Sampling index: %d\n", drafts[keeps[0]].i_batch_tgt[i_dft]);


            LOG("last: %s\n", LOG_TOKENS_TOSTR_PRETTY(ctx_tgt, ctx_sampling->prev).c_str());


            LOG("Sampled token: %d ('%s'), n_past_tgt: %d, current_run.n_past_tgt + i_dft: %d, drafts[keeps[0]].i_batch_tgt[i_dft]: %d\n", id, token_str.c_str(), n_past_tgt, current_run.n_past_tgt + i_dft, drafts[keeps[0]].i_batch_tgt[i_dft]);


            if (current_run.n_past_tgt + i_dft == n_past_tgt-1) {
                any_match = true;
                ++n_predict;
                if (current_run.speculative) {
                    n_accept++;
                }

                if (has_run_first_token) {
                    if (first_token) {
                        ttft = ggml_time_us() - ttft;
                        LOG("\nTTFT: %ld\n", ttft);
                        first_token = false;
                    } else {
                        inter_token_times.push_back(ggml_time_us() - itt_start);
                    }

                    itt_start = ggml_time_us();
                }
                llama_sampling_accept(ctx_sampling, ctx_tgt, id, true);

                // Root of WORLD
                LOG("Accepting token %d ('%s'), n_past_tgt: %d\n", id, token_str.c_str(), n_past_tgt);
                generated.push_back(id);
                if (llama_node_id(ctx_tgt) == 0) {
                    if (!params.use_color) {
                        printf("%s", token_str.c_str());
//                        fprintf(stderr, "%s", token_str.c_str());
                        fflush(stdout);
//                        fflush(stderr);
                    }

                }
            }

            // Switch back to target pipeline only
            llama_swap_comm(ctx_tgt);
            LOG("Swapped comm to target only, id %d\n", llama_node_id(ctx_tgt));



            if (id == llama_token_eos(model_tgt)) {
                has_eos = true;
            }





            if (!current_run.speculative) {
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
                    if (current_run.n_past_tgt + i_dft == n_past_tgt-1) {
                        ++n_accept;
                        ++n_past_tgt;
                        ++n_past_dft;
                    }
                    ++i_dft;
                    if (params.use_color) {
                        // Color token according to its origin sequence
                        printf("\u001b[%dm%s\u001b[37m", (36 - s_keep % 6), token_str.c_str());
                        fflush(stdout);
                    }
                    if (current_run.speculative && current_run.n_past_tgt + i_dft < n_past_tgt) {
                        continue;
                    }
                }
            }

            if (params.use_color) {
                printf("%s", token_str.c_str());
            }
            fflush(stdout);

        }



        if (llama_node_id(ctx_tgt) < 0) {
            LOG("the sampled target token (%d, '%s') did not match, or we ran out of drafted tokens\n", id, token_str.c_str());

        }

        if (!any_match) {
            if (current_run.speculative) {
//                fprintf(stderr, "\nNo match, pushing seq offset %d to free seq offsets\n", current_run.seq_offset);
//                fflush(stderr);
                free_sequence_offsets.push_back(current_run.seq_offset);
            }
//            fprintf(stderr, "No match\n");
            continue;
        }

        check_for_cancel(ctx_tgt, n_past_tgt, tgt_cgraphs, generated, n_seq_dft);


        // TODO: simplify
        if (current_run.speculative){
            LOG("keeping sequence %d, n_past_tgt = %d, n_past_dft = %d, current_run.n_past_tgt = %d, current_run.n_past_dft = %d\n", s_keep+current_run.seq_offset, n_past_tgt, n_past_dft, current_run.n_past_tgt, current_run.n_past_dft);

            for (int i = 0; i < n_seq_dft; i++) {

                llama_kv_cache_seq_rm  (ctx_tgt, i+current_run.seq_offset, n_past_tgt, -1);
                llama_kv_cache_seq_rm  (ctx_dft, i+current_run.seq_offset, n_past_dft, -1);


            }

            llama_kv_cache_seq_rm  (ctx_tgt, 0, current_run.n_past_tgt+1, n_past_tgt);
            llama_kv_cache_seq_rm  (ctx_dft, 0, current_run.n_past_dft+1, n_past_dft);

            llama_kv_cache_seq_cp  (ctx_tgt, s_keep+current_run.seq_offset, 0, current_run.n_past_tgt+1, n_past_tgt);
            llama_kv_cache_seq_cp  (ctx_dft, s_keep+current_run.seq_offset, 0, current_run.n_past_dft+1, n_past_dft);


            for (int i = 1; i <= max_seq; i++) {

                llama_kv_cache_seq_rm(ctx_tgt, i, current_run.n_past_tgt+1, n_past_tgt);
                llama_kv_cache_seq_rm(ctx_dft, i, current_run.n_past_dft+1, n_past_dft);

                llama_kv_cache_seq_cp(ctx_tgt, 0, i, current_run.n_past_tgt+1, n_past_tgt);
                llama_kv_cache_seq_cp(ctx_dft, 0, i, current_run.n_past_dft+1, n_past_dft);


            }




            if (llama_node_id(ctx_tgt) == 0) {
//                llama_kv_cache_view_update(ctx_tgt, &kvc_view);
//                LOG("Done keeping sequence, new tgt cache layout:\n%s", dump_kv_cache_view_seqs(kvc_view, 1).c_str());
            } else {
//                llama_kv_cache_view_update(ctx_dft, &kvc_view_dft);
//                LOG("Done keeping sequence, new dft cache layout:\n%s", dump_kv_cache_view_seqs(kvc_view_dft, 1).c_str());
            }




        }

        begin_non_spec_run(params, n_seq_dft, ctx_tgt, max_seq, drafts, id, batch_id, n_past_tgt, n_past_dft, tgt_cgraphs,
                           kvc_view);

        begin_non_spec_run(params, n_seq_dft, ctx_dft, max_seq, drafts, id, batch_id, n_past_dft, n_past_dft, dft_cgraphs,
                           kvc_view_dft);

        if (!has_run_first_token) {

            has_run_first_token = true;
            first_token = true;
        }

        seq_async_run dft_run = dft_cgraphs.back();
        dft_cgraphs.pop_back();
        llama_finish_async_decode(*ctx_dft, dft_run.batch, dft_run.cgraph);


        spec_past_tgt = n_past_tgt;
        spec_past_dft = n_past_dft;


        if (!current_run.speculative) {
            if (free_sequence_offsets.empty()) {
                continue;
            }
            current_run.seq_offset = free_sequence_offsets.front();
//            if (llama_node_id(ctx_tgt) == 0) {
//                fprintf(stderr, "Popping %d from seq offsets for spec run\n", current_run.seq_offset);
//                fflush(stderr);
//            }
            free_sequence_offsets.pop_front();
        }


//        bool is_waiting = false;

        run_speculation_loop(params, p_accept, ctx_tgt, ctx_dft, max_seq, batch_tgt, n_predict, n_past_tgt, n_past_dft,
                             has_eos, ctx_sampling,
                             spec_past_tgt, spec_past_dft, first_run, free_sequence_offsets, batch_id, batch_dft,
                             n_drafted, drafts, tgt_cgraphs, current_run, kvc_view_dft, id);


        if (n_predict > params.n_predict || has_eos) {
            break;
        }




    }

    auto t_dec_end = ggml_time_us();

    uint64_t avg_itt = 0;
    for (auto latency : inter_token_times) {
        avg_itt += latency;
    }

    avg_itt = avg_itt / inter_token_times.size();

    LOG_TEE("\n\n");

    LOG_TEE("encoded %4d tokens in %8.3f seconds, speed: %8.3f t/s\n", n_input,   (t_enc_end - t_enc_start) / 1e6f, inp.size() / ((t_enc_end - t_enc_start) / 1e6f));
    LOG_TEE("decoded %4d tokens in %8.3f seconds, speed: %8.3f t/s\n", n_predict, (t_dec_end - t_dec_start) / 1e6f, n_predict  / ((t_dec_end - t_dec_start) / 1e6f));
    LOG_TEE("Average inter-token latency: %f seconds\n", avg_itt / 1e6f);
    LOG_TEE("Time-to-first-token: %f seconds\n", ttft / 1e6f);
    
    
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

    if (llama_node_id(ctx_tgt) == 0) {
        for (size_t i = tgt_cgraphs.size() - 1; i >= 0; i--) {
            const auto &run = tgt_cgraphs[i];
            llama_finish_async_decode(*ctx_tgt, run.batch, run.cgraph);
        }
    }

    for (size_t i = dft_cgraphs.size()-1; i >= 0; i--) {
        const auto& run = dft_cgraphs[i];
        llama_finish_async_decode(*ctx_dft, run.batch, run.cgraph);
    }

    llama_free(ctx_tgt);
    llama_free_model(model_tgt);

    llama_free(ctx_dft);
    llama_free_model(model_dft);

    llama_backend_free();

    fprintf(stderr, "\n\n");

    return 0;
}

void
run_speculation_loop(const gpt_params &params, const float p_accept, llama_context *ctx_tgt, llama_context *ctx_dft,
                     const int max_seq, llama_batch &batch_tgt, int n_predict, int n_past_tgt, int n_past_dft,
                     bool has_eos, llama_sampling_context *ctx_sampling, int &spec_past_tgt, int &spec_past_dft,
                     bool & first_run, std::deque<int> &free_sequence_offsets, int32_t &batch_id, llama_batch &batch_dft,
                     int &n_drafted, std::vector<seq_draft> &drafts, std::deque<struct seq_async_run> &tgt_cgraphs,
                     seq_async_run &current_run, llama_kv_cache_view &kvc_view_dft, llama_token &id) {
    bool is_waiting = llama_mpi_iprobe(ctx_tgt);
    llama_swap_comm(ctx_tgt);
    llama_sync_token(ctx_tgt, reinterpret_cast<llama_token *>(&is_waiting), 0);
    llama_swap_comm(ctx_tgt);


    if (is_waiting) {
//            fprintf(stderr, "\nIs waiting, pushing seq offset %d to free seq offsets\n", current_run.seq_offset);
//            fflush(stderr);
        free_sequence_offsets.push_back(current_run.seq_offset);
    }
    int iter = 0;
    while((!is_waiting && (p_accept + iter * params.p_recovery) < 1.0)) {





        int orig_offset = current_run.seq_offset;
        bool should_run_spec = true;
        std::deque<int> checked_offsets;
        do {
            should_run_spec = true;
            for (const auto &r: tgt_cgraphs) {
                if (r.seq_offset == current_run.seq_offset && r.speculative) {
                    checked_offsets.push_back(current_run.seq_offset);

                    should_run_spec = false;
                    if (!free_sequence_offsets.empty()) {
                        current_run.seq_offset = free_sequence_offsets.front();
                        free_sequence_offsets.pop_front();

                    }
                    break;
                }
            }
        } while (!should_run_spec && !free_sequence_offsets.empty());

        if (!should_run_spec) {
            LOG("Ending spec because no available offsets\n");
            break;
        }
//            if (llama_node_id(ctx_tgt) == 0) {
//                fprintf(stderr, "\nErasing seq offset %d from free seq offsets\n", current_run.seq_offset);
//                fflush(stderr);
//            }
        auto it = std::find(free_sequence_offsets.begin(), free_sequence_offsets.end(), current_run.seq_offset);
        if (it != free_sequence_offsets.end()) {
            free_sequence_offsets.erase(it);
        }


        if (start_async_spec_run(params, ctx_tgt, ctx_dft, free_sequence_offsets, max_seq,
                                 batch_tgt, n_predict, n_past_tgt, n_past_dft, has_eos, ctx_sampling,
                                 tgt_cgraphs,
                                 current_run, spec_past_tgt, spec_past_dft, first_run, orig_offset,
                                 batch_id, batch_dft, n_drafted, drafts, id, kvc_view_dft, iter)) {
            LOG("Ending spec run because returned true\n");
            break;
        }

        is_waiting = llama_mpi_iprobe(ctx_tgt);
        llama_swap_comm(ctx_tgt);
        llama_sync_token(ctx_tgt, reinterpret_cast<llama_token *>(&is_waiting), 0);
        llama_swap_comm(ctx_tgt);
        first_run = false;

        iter++;
//            break;

    }
}

void begin_non_spec_run(const gpt_params &params, const int n_seq_dft, llama_context *ctx, const int max_seq,
                        const std::vector<seq_draft> &drafts, llama_token id, int32_t &batch_id, int &n_past,
                        int n_past_dft,
                        std::deque<struct seq_async_run> &dft_cgraphs, llama_kv_cache_view &kvc_view) {

    std::vector<seq_draft> non_spec_drafts = std::vector<seq_draft>(n_seq_dft);
    for (int s = 0; s < n_seq_dft; ++s) {
        non_spec_drafts[s].ctx_sampling = llama_sampling_init(params.sparams);
        llama_sampling_cp(drafts[s].ctx_sampling, drafts[s].ctx_sampling);
        non_spec_drafts[s].i_batch_tgt = std::vector<int>(1,0);
        non_spec_drafts[s].i_batch_dft = drafts[s].i_batch_dft;
        non_spec_drafts[s].tokens = std::vector<llama_token>(1, id);
        non_spec_drafts[s].active = drafts[s].active;
        non_spec_drafts[s].drafting = drafts[s].drafting;
        non_spec_drafts[s].skip = drafts[s].skip;
        non_spec_drafts[s].prefix_tokens = std::vector<llama_token>(0);
    }

    llama_batch async_batch = llama_batch_init(params.n_ctx, 0, max_seq + 1);

    llama_batch_clear(async_batch);

    llama_batch_add(async_batch, id, n_past, {0}, true);

    begin_async_run(params.sparams, n_seq_dft, ctx, max_seq, n_past_dft,
                    non_spec_drafts, dft_cgraphs, batch_id, n_past, kvc_view, false, async_batch, n_past+1, n_past, 0);

    n_past++;

}

bool start_async_spec_run(const gpt_params &params, llama_context *ctx_tgt, llama_context *ctx_dft,
                          std::deque<int> &free_sequence_offsets, int max_seq, llama_batch &batch_tgt, int n_predict,
                          int prefix_n_past, int n_past_dft, bool has_eos, llama_sampling_context *ctx_sampling,
                          std::deque<struct seq_async_run> &tgt_cgraphs, const seq_async_run &current_run,
                          int &spec_past_tgt, int &spec_past_dft, int first_run, int orig_offset, int32_t &batch_id,
                          llama_batch &batch_dft, int &n_drafted, std::vector<seq_draft> &drafts, llama_token &id,
                          llama_kv_cache_view &kvc, const int iter) {
    LOG("Doing speculative run, seq_offset = %d, spec_past_tgt = %d, spec_past_dft = %d, prefix_n_past = %d, n_past_dft = %d\n",
        current_run.seq_offset, spec_past_tgt, spec_past_dft, prefix_n_past, n_past_dft);

    for (int i = 0; i < params.n_parallel; i++) {

        llama_kv_cache_seq_rm(ctx_tgt, i + current_run.seq_offset, (first_run) ? prefix_n_past : prefix_n_past - 1, -1);
        llama_kv_cache_seq_rm(ctx_dft, i + current_run.seq_offset, (first_run) ? n_past_dft : n_past_dft - 1, -1);

        LOG("Copying tgt sequence %d to %d from positions %d to %d\n", (first_run) ? 0 : orig_offset,
            i + current_run.seq_offset, prefix_n_past, spec_past_tgt);

        llama_kv_cache_seq_cp(ctx_tgt, (first_run) ? 0 : orig_offset, i + current_run.seq_offset, (first_run) ? prefix_n_past : prefix_n_past - 1,
                              spec_past_tgt+1);
        llama_kv_cache_seq_cp(ctx_dft, (first_run) ? 0 : orig_offset, i + current_run.seq_offset, (first_run) ? n_past_dft : n_past_dft - 1,
                              spec_past_dft+1);


    }


    llama_batch_clear(batch_tgt);

    for (int s = 0; s < params.n_parallel; ++s) {
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


    llama_batch_clear(batch_dft);


    if (llama_node_id(ctx_dft) == 0) {
//            llama_kv_cache_view_update(ctx_dft, &kvc);
//        LOG("Draft KV cache view:\n%s\n", dump_kv_cache_view_seqs(kvc, 1).c_str());
    }


    llama_sampling_cp(ctx_sampling, drafts[0].ctx_sampling);

    int n_seq_cur  = 0;
    int max_ran_seq = 0;
    int n_past_cur = spec_past_dft;

    for (int s = 0; s < params.n_parallel; ++s) {
        drafts[s].skip = true;
        drafts[s].active = false;
        drafts[s].drafting = false;
    }


    drafts[0].active = true;
    drafts[0].drafting = true;
    drafts[0].skip = false;

    drafts[0].i_batch_dft = 0;


    // sample n_draft tokens from the draft model using tree-based sampling
    for (int i = 0; i < params.n_draft; ++i) {
        batch_dft.n_tokens = 0;



        for (int s = 0; s <= max_ran_seq; ++s) {
            if (!drafts[s].drafting || drafts[s].skip) {
                continue;
            }



            // Swap back to pipeline roots
            llama_swap_comm(ctx_dft);
            LOG("Swapped comm to pipeline roots, id %d\n", llama_node_id(ctx_dft));

            llama_sync_token(ctx_dft, &(drafts[s].i_batch_dft), 1);

            llama_sampling_sample(drafts[s].ctx_sampling, ctx_dft, nullptr, drafts[s].i_batch_dft);

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
                for (int k = 0; k < std::min(params.n_parallel, (int) cur_p.size()); ++k) {
                    LOG(" - draft candidate %3d for seq %3d, pos %3d: %6d (%8.3f) '%s'\n",
                        k, s+current_run.seq_offset, i+spec_past_dft, cur_p[k].id, cur_p[k].p, llama_token_to_piece(ctx_dft, cur_p[k].id).c_str());
                }
            }


            if (cur_p[0].p < params.p_accept + params.p_recovery * iter) {
                LOG("stopping drafting for seq %3d, probability too low: %.3f < %.3f\n", s, cur_p[0].p,
                    params.p_accept);
                drafts[s].drafting = false;
                continue;
            }


            std::vector<int> sa(1, s);

            // attempt to split the branch if the probability is high enough
            for (int f = 1; f < 8; ++f) {
                if (n_seq_cur < params.n_parallel - 1 && cur_p[f].p > params.p_split + params.p_recovery * iter) {
                    n_seq_cur++;
                    LOG("splitting seq %3d into %3d\n", s, n_seq_cur);


                    LOG("Removing dft sequence %d from positions %d to %d\n", n_seq_cur + current_run.seq_offset, n_past_dft, n_past_cur);

                    llama_kv_cache_seq_rm(ctx_dft, n_seq_cur + current_run.seq_offset, n_past_dft, n_past_cur);

                    LOG("Copying dft sequence %d to %d from positions %d to %d\n", s + current_run.seq_offset, n_seq_cur + current_run.seq_offset, n_past_dft, n_past_cur);

                    llama_kv_cache_seq_cp(ctx_dft, s + current_run.seq_offset, n_seq_cur + current_run.seq_offset, n_past_dft, n_past_cur);

                    // all previous tokens from this branch are now also part of the new branch
                    for (int t = 0; t < batch_tgt.n_tokens; ++t) {
                        for (int p = 0; p < batch_tgt.n_seq_id[t]; ++p) {
                            if (batch_tgt.seq_id[t][p] == s + current_run.seq_offset) {
                                batch_tgt.seq_id[t][batch_tgt.n_seq_id[t]] = n_seq_cur + current_run.seq_offset;
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
                    s + current_run.seq_offset, spec_past_tgt + i, batch_tgt.n_tokens);
                llama_batch_add(batch_tgt, id, spec_past_tgt + i, {s + current_run.seq_offset}, true);

                // add the token to the batch for batched decoding with the draft model
                drafts[s].i_batch_dft = batch_dft.n_tokens;

                LOG("Adding drafted token %d to dft, sequence %d, position %d\n", id, s + current_run.seq_offset, n_past_cur);

                llama_batch_add(batch_dft, id, n_past_cur, {s + current_run.seq_offset}, true);

                if (batch_tgt.n_tokens > params.n_draft) {
                    drafts[s].drafting = false;
                }
            }
        }

        // no sequence is drafting anymore
        if (batch_dft.n_tokens == 0) {
            break;
        }

        // evaluate the drafted tokens on the draft model
        LOG("Running synchronous draft decode while still drafting\n");
        LOG("dft batch: %s\n", LOG_BATCH_TOSTR_PRETTY(ctx_dft, batch_dft).c_str());
        llama_decode(ctx_dft, batch_dft);
        ++n_past_cur;
        ++n_drafted;

        max_ran_seq = n_seq_cur;

        llama_batch_clear(batch_dft);

        if (batch_tgt.n_tokens > params.n_draft) {
            break;
        }
    }

    // no sequence is drafting anymore
    if (batch_dft.n_tokens != 0) {
        // evaluate the drafted tokens on the draft model
        LOG("Running synchronous draft decode when no seqs drafting\n");
        LOG("dft batch: %s\n", LOG_BATCH_TOSTR_PRETTY(ctx_dft, batch_dft).c_str());
        llama_decode(ctx_dft, batch_dft);

    }







    // evaluate the target model on the drafted tokens
    {
//            llama_kv_cache_seq_keep(ctx_tgt, 0); // Needed to get to "Here's the code:"





        if (batch_tgt.n_tokens == 0) {
//            fprintf(stderr, "\nNo tgt tokens, pushing seq offset %d to free seq offsets\n", current_run.seq_offset);
//            fflush(stderr);
            free_sequence_offsets.push_back(current_run.seq_offset);
            return true;
        }

//        bool is_waiting = llama_mpi_iprobe(ctx_tgt);
//        llama_swap_comm(ctx_tgt);
//        llama_sync_token(ctx_tgt, reinterpret_cast<llama_token *>(&is_waiting), 0);
//        llama_swap_comm(ctx_tgt);
//
//        if (is_waiting) {
//            free_sequence_offsets.push_back(current_run.seq_offset);
//            return true;
//        }

        size_t max_draft_tokens = 0;

        for (int s = 0; s < params.n_parallel; ++s) {
            if (!drafts[s].active) {
                continue;
            }

            drafts[s].tokens.erase(drafts[s].tokens.begin());
            max_draft_tokens = std::max(max_draft_tokens, drafts[s].tokens.size());
            //drafts[s].tokens.erase(drafts[s].tokens.begin());
        }

        if (first_run) {
            ++n_drafted;
        }

        begin_async_run(params.sparams, params.n_parallel, ctx_tgt, max_seq, n_past_dft, drafts, tgt_cgraphs,
                        batch_id, spec_past_tgt, kvc, true, batch_tgt, spec_past_tgt + drafts[0].tokens.size(), prefix_n_past, current_run.seq_offset);

        spec_past_tgt += drafts[0].tokens.size();
        spec_past_dft += drafts[0].tokens.size();
        id = drafts[0].tokens.back();
        first_run = false;

//        LOG("Beginning tgt spec run, run.prefix_n_past=%d, run.prefix_n_past_tgt=%d, run.n_past_dft=%d, run.n_past_max=%d, new spec_past_tgt=%d, new spec_past_dft=%d, new id=%d\n",
//            run.prefix_n_past, run.prefix_n_past_tgt, run.n_past_dft, run.n_past_max, spec_past_tgt, spec_past_dft, id
//        );

    }

    return false;


}

void begin_async_run(const llama_sampling_params& sparams, const int n_seq_dft,
                     llama_context *ctx_tgt, const int max_seq,
                     int n_past_dft, const std::vector<seq_draft> &drafts,
                     std::deque<struct seq_async_run> &tgt_cgraphs,
                     int32_t &batch_id, int &n_past, llama_kv_cache_view &kvc_view,
                     const bool is_spec, llama_batch batch, const int n_past_max, const int prefix_n_past, const int seq_offset) {
    batch_id++;


    LOG("Beginning async decode, batch id = %d\n", batch_id);





    // batch_tgt.n_tokens = 1


    struct seq_async_run run;
    run.seq_offset = seq_offset;
    run.batch = llama_batch_init(1028, 0, max_seq);
    run.batch.batch_id = batch_id;
    run.batch.n_tokens = batch.n_tokens;
    for (int i = 0; i < batch.n_tokens; i++) {
        run.batch.n_seq_id[i] = batch.n_seq_id[i];
        int cur_n_seqs = 0;
        for (int j = 0; j < run.batch.n_seq_id[i]; j++) {
            run.batch.seq_id[i][j] = batch.seq_id[i][j];
        }
        run.batch.token[i] = batch.token[i];
        run.batch.pos[i] = batch.pos[i];
        run.batch.logits[i] = batch.logits[i];
    }
    run.batch.batch_id = batch_id;
    run.canceled = false;
    run.s_keep = 0;
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



    LOG("target async batch: %s\n, batch_id = %d\n", LOG_BATCH_TOSTR_PRETTY(ctx_tgt, run.batch).c_str(),
        batch_id);

    run.drafts = std::vector<seq_draft>(n_seq_dft);
    for (int s = 0; s < n_seq_dft; ++s) {
        run.drafts[s].ctx_sampling = llama_sampling_init(sparams);
        llama_sampling_cp(drafts[s].ctx_sampling, run.drafts[s].ctx_sampling);
        run.drafts[s].i_batch_tgt = drafts[s].i_batch_tgt;
        run.drafts[s].i_batch_dft = drafts[s].i_batch_dft;
        run.drafts[s].tokens = drafts[s].tokens;
        run.drafts[s].active = drafts[s].active;
        run.drafts[s].drafting = drafts[s].drafting;
        run.drafts[s].skip = drafts[s].skip;
        run.drafts[s].prefix_tokens = drafts[s].prefix_tokens;
    }
    run.n_past_tgt = n_past;
    run.prefix_n_past_tgt = prefix_n_past;
    run.n_past_max = n_past_max;
    run.n_past_dft = n_past_dft;
    run.speculative = is_spec;

    if (!is_spec) {
        for (int i = 0; i <= max_seq; i++) {
            llama_kv_cache_seq_rm(ctx_tgt, i, n_past, n_past + 1);
        }
    } else {
        for (int i = 0; i < n_seq_dft; i++) {
            llama_kv_cache_seq_rm(ctx_tgt, i+seq_offset, n_past, n_past + 1);
        }
    }
    run.cgraph = llama_start_async_decode(*ctx_tgt, run.batch);
    tgt_cgraphs.push_front(run);

    if (!is_spec) {
        for (int i = 1; i <= max_seq; i++) {
            llama_kv_cache_seq_cp(ctx_tgt, 0, i, n_past, n_past + 1);
        }
    }

    if (llama_node_id(ctx_tgt) == 0) {
//        llama_kv_cache_view_update(ctx_tgt, &kvc_view);
//        LOG("Done running non-spec, cache view:\n%s", dump_kv_cache_view_seqs(kvc_view, 1).c_str());
//        printf("\nBeginning async run, batch id: %d, batch: %s\n", run.batch.batch_id, LOG_BATCH_TOSTR_PRETTY(ctx_tgt, run.batch).c_str());
    }
}

void check_for_cancel(llama_context *ctx_tgt, int n_past_tgt, std::deque<struct seq_async_run> &tgt_cgraphs,
                      std::vector<llama_token> &generated, const int n_seq_dft) {
    std::vector<int> canceled_batches;
    for (auto &run : tgt_cgraphs) {
        if(!run.canceled) {
            bool correct_prefix = true;

            if (run.speculative && n_past_tgt >= run.prefix_n_past_tgt) {
                for (int draft_id = n_seq_dft - 1; draft_id >= 0; draft_id--) {
                    if (!run.drafts[draft_id].tokens.empty()) {
                        correct_prefix = true;
                    } else {
                        continue;
                    }
                    size_t draft_index = 0;
                    int prev_token = -1;
                    int prev_gen_token = -1;
                    std::vector<llama_token> concat_tokens = run.drafts[draft_id].prefix_tokens;
                    concat_tokens.insert(concat_tokens.end(), run.drafts[draft_id].tokens.begin(),
                                         run.drafts[draft_id].tokens.end());


                    LOG("Prefix tokens: %s\n", LOG_TOKENS_TOSTR_PRETTY(ctx_tgt, run.drafts[draft_id].prefix_tokens).c_str());

                    LOG("Concat tokens: %s\n", LOG_TOKENS_TOSTR_PRETTY(ctx_tgt, concat_tokens).c_str());


                    size_t index = run.prefix_n_past_tgt + draft_index;
                    LOG("Looping over run starting at gen index %zu, draft index %zu, prefix_n_past_tgt %d, n_past_tgt %d, generated size %zu\n",
                        index, draft_index, run.prefix_n_past_tgt, n_past_tgt, generated.size());
                    while (index < generated.size() && draft_index < concat_tokens.size() &&
                           generated.size() > (size_t) run.prefix_n_past_tgt) {
                        LOG("Checking draft at index %zu and generated index %zu\n", draft_index, index);
                        if (generated.at(index) != concat_tokens[draft_index]) {
                            LOG("Found non-matching prefix at generated index %zu, draft index %zu, gen token %d, draft token %d, prev draft token %d, prev gen token %d\n",
                                index, draft_index, generated.at(index), concat_tokens[draft_index], prev_token,
                                prev_gen_token);
                            correct_prefix = false;
                            break;
                        }
                        prev_token = concat_tokens[draft_index];
                        prev_gen_token = generated[index];
                        draft_index++;
                        index = run.prefix_n_past_tgt + draft_index;
                    }
                    if (correct_prefix) {
                        run.s_keep = draft_id;
                    }
                }
            }


            if (run.n_past_max < n_past_tgt || !correct_prefix) {
                LOG("Cancelling batch ID %d, run.npast_max %d, run.n_past_tgt %d, n_past_tgt %d, run_speculative %d, tokens[0] %d, generated: %d, generated index: %zu\n",
                    run.batch.batch_id, run.n_past_max, run.n_past_tgt, n_past_tgt, run.speculative,
                    run.drafts[0].tokens[0], (n_past_tgt < run.n_past_tgt) ? -1 : generated.at(
                        generated.size() - (n_past_tgt - run.n_past_tgt + 1)),
                    generated.size() - (n_past_tgt - run.n_past_tgt + 1));

                if (run.speculative) {
                    // TODO put these in a vector so they are transmitted in a burst
                    canceled_batches.push_back(run.batch.batch_id);
                    for (int i = 0; i < n_seq_dft; i++) {

//                        llama_kv_cache_seq_rm  (ctx_tgt, i+run.seq_offset, run.n_past_tgt, -1);


                    }
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
