The models used in our paper can be downloaded from the following links (compatible speculative models are listed as children of their respective target models):

- [Dolphin 70B](https://huggingface.co/TheBloke/Dolphin-2.2-70B-GGUF)
  - [TinyLlama OpenOrca 1.1B](https://huggingface.co/TheBloke/TinyLlama-1.1B-1T-OpenOrca-GGUF)
  - [Orca2](https://huggingface.co/TheBloke/Orca-2-7B-GGUF)
- [Goliath 120B](https://huggingface.co/TheBloke/goliath-120b-GGUF)
  - [XWinLM 7B](https://huggingface.co/TheBloke/Xwin-LM-7B-V0.2-GGUF)
  - [XWinLM 13B](https://huggingface.co/TheBloke/Xwin-LM-13B-v0.2-GGUF)
- [Falcon 180B](https://huggingface.co/TheBloke/Falcon-180B-GGUF)
  - [Falcon 7B](https://huggingface.co/maddes8cht/tiiuae-falcon-7b-gguf)
  - [Falcon 40B](https://huggingface.co/maddes8cht/tiiuae-falcon-40b-gguf)
 
    
  The above links host files for several different quantization levels. The specific quantizations used can be found in Table 1 of our paper.

  When running PipeInfer, one must pass the large, target model as the `-m` parameter, and the smaller speculative model as the `-md` parameter.
  An example for running Dolphin 70B with a speculative model of TinyLlama, given that each model is downloaded to a `models/` folder:

  ```bash
  -m models/models/dolphin-2.2-70b.Q3_K_M.gguf -md models/tinyllama-1.1b-1t-openorca.Q4_K_M.gguf
  ```


  
