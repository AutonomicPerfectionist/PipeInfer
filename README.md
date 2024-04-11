# PipeInfer: Accelerating LLM Inference using Asynchronous Pipelined Speculation

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)

Based on llama.cpp, inference of [LLaMA](https://arxiv.org/abs/2302.13971) model in pure C/C++.
----

<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#description">Description</a>
    </li>
    <li>
      <a href="#usage">Usage</a>
      <ul>
        <li><a href="#get-the-code">Get the Code</a></li>
        <li><a href="#build">Build</a></li>
        <li><a href="#run">Run</a></li>
      </ul>
    </li>
    <li><a href="#tuning-pipeinfer">Tuning PipeInfer</a></li>
  </ol>
</details>

## Description

PipeInfer is a modification of [llama.cpp](https://github.com/ggerganov/llama.cpp) that is designed
for acceleration of inference across multi-node clusters using pipelined asynchronous speculation.

- Supports Llama/Llama2, Falcon, Baichuan, Starcoder, Bloom, and other architectures.
- Based on llama.cpp commit [9656026](https://github.com/ggerganov/llama.cpp/commit/9656026b53236ed7328458269c4c798dd50ac8d1)
- See progress on porting to the latest version of llama.cpp [here](https://github.com/AutonomicPerfectionist/llama.cpp/tree/mpi-heterogenous)

**Supported platforms:**

- [X] Mac OS
- [X] Linux
- [X] Docker

**Supported models:**

- [X] LLaMA 🦙
- [x] LLaMA 2 🦙🦙
- [X] Falcon
- [X] [Alpaca](https://github.com/ggerganov/llama.cpp#instruction-mode-with-alpaca)
- [X] [GPT4All](https://github.com/ggerganov/llama.cpp#using-gpt4all)
- [X] [Chinese LLaMA / Alpaca](https://github.com/ymcui/Chinese-LLaMA-Alpaca) and [Chinese LLaMA-2 / Alpaca-2](https://github.com/ymcui/Chinese-LLaMA-Alpaca-2)
- [X] [Vigogne (French)](https://github.com/bofenghuang/vigogne)
- [X] [Vicuna](https://github.com/ggerganov/llama.cpp/discussions/643#discussioncomment-5533894)
- [X] [Koala](https://bair.berkeley.edu/blog/2023/04/03/koala/)
- [X] [OpenBuddy 🐶 (Multilingual)](https://github.com/OpenBuddy/OpenBuddy)
- [X] [Pygmalion/Metharme](#using-pygmalion-7b--metharme-7b)
- [X] [WizardLM](https://github.com/nlpxucan/WizardLM)
- [X] [Baichuan 1 & 2](https://huggingface.co/models?search=baichuan-inc/Baichuan) + [derivations](https://huggingface.co/hiyouga/baichuan-7b-sft)
- [X] [Aquila 1 & 2](https://huggingface.co/models?search=BAAI/Aquila)
- [X] [Starcoder models](https://github.com/ggerganov/llama.cpp/pull/3187)
- [X] [Mistral AI v0.1](https://huggingface.co/mistralai/Mistral-7B-v0.1)
- [X] [Refact](https://huggingface.co/smallcloudai/Refact-1_6B-fim)
- [X] [Persimmon 8B](https://github.com/ggerganov/llama.cpp/pull/3410)
- [X] [MPT](https://github.com/ggerganov/llama.cpp/pull/3417)
- [X] [Bloom](https://github.com/ggerganov/llama.cpp/pull/3553)
- [X] [StableLM-3b-4e1t](https://github.com/ggerganov/llama.cpp/pull/3586)


---
## Usage

PipeInfer is implemented in the `speculative` example binary. To run PipeInfer follow these steps.

### Get the Code

```bash
git clone https://github.com/AutonomicPerfectionist/PipeInfer
cd PipeInfer
```

### Install Dependencies

Make sure you have Make or CMake installed, as well as an MPI implementation and compiler.
On Debian-based systems these can be installed with the following:

```bash
$ sudo apt install build-essential cmake openmpi-bin libopenmpi-dev
```

### Build

To build PipeInfer you have two different options.

- Using `make`:
  - On Linux or MacOS:

      ```bash
      make speculative CC=mpicc CXX=mpicxx LLAMA_MPI=ON -j
      ```

- Using `CMake`:

    ```bash
    mkdir build
    cd build
    cmake .. -DCMAKE_C_COMPILER=mpicc -DCMAKE_CXX_COMPILER=mpicxx -DLLAMA_MPI=1
    cmake --build . --target speculative --config Release
    ```


### Download Models
PipeInfer requires two models to be downloaded: the large target model
and a smaller speculative model. Both models must have compatible vocabulary,
minor differences are allowed but may cause performance degradation. The models
used in our paper can be downloaded from the following links:

- [Dolphin 70B](https://huggingface.co/TheBloke/Dolphin-2.2-70B-GGUF)
- [TinyLlama OpenOrca 1.1B](https://huggingface.co/TheBloke/TinyLlama-1.1B-1T-OpenOrca-GGUF)
- [Orca2](https://huggingface.co/TheBloke/Orca-2-7B-GGUF)
- [Goliath 120B](https://huggingface.co/TheBloke/goliath-120b-GGUF)
- [XWinLM 7B](https://huggingface.co/TheBloke/Xwin-LM-7B-V0.2-GGUF)
- [XWinLM 13B](https://huggingface.co/TheBloke/Xwin-LM-13B-v0.2-GGUF)
- [Falcon 180B](https://huggingface.co/TheBloke/Falcon-180B-GGUF)
- [Falcon 7B](https://huggingface.co/maddes8cht/tiiuae-falcon-7b-gguf)
- [Falcon 40B](https://huggingface.co/maddes8cht/tiiuae-falcon-40b-gguf)




### Run
To run PipeInfer, make sure you have MPI configured on all nodes, and that the head node can SSH into all other nodes.
It is recommended to place the model files on shared storage to avoid replicating them to each node. We have had success
using a dedicated NFS file server. By default `llama.cpp` loads the models by `mmap`'ing them, and only
the tensors needed by the layers assigned to a node will be faulted in. For multi-socket systems, it is recommended to clear the Linux file cache
whenever the layer allocations are changed, as otherwise the tensors may be kept on the wrong NUMA node.

This can be done with the following command:

```bash
$ echo 3 | sudo tee /proc/sys/vm/drop_caches
```

Alternatively, using the `--no-mmap` switch when running PipeInfer/llama.cpp will read the file contents into RAM directly, bypassing the file cache.
Note that doing so forces the entire model to be loaded into RAM, potentially exceeding the memory capacity of the node.

To execute PipeInfer on a cluster, ensure that each node has access to the `speculative` binary and the model file. Like any other MPI program,
it can be launched with `mpirun` or through job managers like Slurm. PipeInfer also requires the layer allocations to be passed in on the command line,
through the `--mpi-layer-split` argument. For each node in the cluster, one must pass in a floating-point value denoting the percentage
of layers the node will handle. When using multiple models, as in the case of PipeInfer, one must also specify how to split the
nodes into two communicators. This split is denoted by a slash `/`. All percentages in the first half are tied to the target model, and percentages after
the slash are tied to the speculative model. An example: if there are 5 nodes total, and we want 4 nodes to handle the target model and one node to handle
the speculative model, with even splits among the target nodes, we would pass the following:

```bash
--mpi-layer-split 0.25,0.25,0.25,0.25/1.0
```

If we wanted to dedicate two nodes to the speculative model instead and only 3 to the target model, we would move the slash so that there are three values
on the left half and two values on the right. The values themselves must also be adjusted. PipeInfer will automatically allocate any leftover layers to the
head node (node 0), but it is best to allocate them explicitly.

An example command to launch PipeInfer across a cluster of 4 nodes is shown below:

```bash
mpirun -hostfile /mnt/cluster/hostsfile --bind-to none \
    /var/llama.cpp/bin/speculative \
    -md /mnt/cluster/models/tinyllama-1.1b-1t-openorca.Q4_K_M.gguf \
    -m /mnt/cluster/models/dolphin-2.1-70b.Q3_K_M.gguf  \
    -e \
    -f /mnt/cluster/llama.cpp/prompts/dolphin.txt \
    -n 128 \
    --mpi-layer-split 0.1,0.4,0.5/1.0 \
    --ignore-eos \
    --temp -1.0 \
    --repeat-last-n 0 \
    --draft 4 \
    -tb 12,32,40,32 \
    -t 6,32,20,32 \
    -c 1024 \
    -pa 0.001 \
    -ps 0.8 \
    --numa \
    -pr 0.4 \
    -pd 0.01 \
    -np 3
```

The confidence cutoff recovery and decay factors are the `-pr` and `-pd` parameters, respectively. The `-pa` parameter is the base acceptance confidence cutoff, and the `-ps` parameter
is the split confidence threshold. The `-np` parameter determines the maximum number of sequences within a sequence partition. The `--draft` parameter determines the maximum number of tokens
in a speculative microbatch. The `-t` and `-tb` parameters set the number of threads to use for single-token and batched inference for each node. Finally, we pass the `--bind-to none` parameter
to `mpirun` to enable each rank to use more than one thread. All other parameters are unmodified from upstream `llama.cpp`; use the `--help` parameter to list all available
parameters and what they do.

## Tuning PipeInfer
PipeInfer offers many tunable parameters that can drastically affect the performance. The most important of these are the `-pa`, `-ps`, `-np`, and `--draft` parameters. These four parameters determine the
average size of a speculative microbatch. An important fact to consider is that the acceptance confidence cutoff (`-pa`) corresponds to how confident the speculative model is in its own
output, *not* how confident it is that it matches the target's output.

Some models, such as Orca 2, have high confidence in their own outputs but low alignment with their target. In such a case,
it is best to set the acceptance confidence cutoff high, such as `0.5`, and the draft length (`--draft`) low, such as `3`.

Conversely, some models exhibit high alignment with their target. In these cases, increasing the draft length slightly can produce
better performance, but it is almost always best to keep the draft length below `8` to maintain low latency and high granularity with regard to cancellation.
Instead, the acceptance confidence cutoff can be lowered significantly, down to `0.1` in our testing with Goliath and XWin 7B. Such a low cutoff
enables PipeInfer to generate as many speculative runs as possible. Keeping the draft length low as well (`3` for Goliath and XWin 7B) enables the system to
generate many small microbatches that can be evaluated with low latency while maintaining a high enough acceptance rate that the pipeline is not thrashed
with flushing operations.

The split confidence threshold and the maximum number of sequences per partition can be tuned to produce a wide and shallow speculative tree; however, we have observed
that narrow and deep trees produce the best overall performance. Keeping the number of sequences per partition low, such as `3` or `4`, allows early inference cancellation
finer control over which sequences to cancel. In the future, it may be possible to extend cancellation to work on individual sequences, but at the moment, we have observed
wide trees to decrease performance universally.

The two adaptive parameters, `-pr` and `-pd`, can be set to allow the system to automatically adjust the speculative tree size and speculation frequency according to the current conditions.
The recovery factor (`-pr`) increases the acceptance confidence cutoff every time a speculative tree is generated, reset upon acceptance of a token in the verification phase.
This has the effect of making it harder to generate speculative trees the longer it's been since a token has been validated. The recovery factor is most useful when the target pipeline is very
deep, and the speculative model diverges fairly quickly. Without this factor, such a system would generate speculative runs even when it is likely the speculative tree has long since diverged from the target model,
wasting CPU cycles and requiring a cancellation later on. We have not observed this factor to increase performance; its only usefulness is in decreasing wasted cycles and transfers.

The decay factor is the inverse, decreasing the acceptance confidence cutoff the longer it's been since any speculative tree was generated, reset upon successful token acceptance
in the verification phase. This factor enables the system to opportunistically generate speculations when it would otherwise be idle, even if they aren't likely to be accepted.
It is best used when the base acceptance cutoff is fairly high, and one observes significant idle time. We have found this factor must be very carefully managed, setting it too high results in
large speculative trees being pushed through the pipeline as soon as there is any idle time, causing an imbalance in evaluation time for each run, therefore causing further idle periods. This can
be mitigated by keeping the draft length low.
