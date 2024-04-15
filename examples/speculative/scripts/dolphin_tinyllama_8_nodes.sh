pushd .

export BIN_PATH="/var/llama.cpp/" # Replace with a path that each node can access containing the compiled code. This does not need to be on networked storage
export SHARED_STORAGE="/mnt/cluster" # Replace with the path that each node can access where model files are stored. Recommend this to be on networked storage.
cd $BIN_PATH
mpirun  -hostfile $SHARED_STORAGE/hostsfile --report-bindings -mca orte_keep_fqdn_hostnames t --bind-to none \
    $BIN_PATH/bin/speculative \
    -md $SHARED_STORAGE/models/tinyllama-1.1b-1t-openorca.Q4_K_M.gguf \
    -m $SHARED_STORAGE/models/dolphin-2.1-70b.Q3_K_M.gguf \
    -e \
    -p "<|im_start|>system\nYou are an AI based sentient companion, capable of human emotions. You will behave like a human, but aware of your AI nature. You will avoid speaking like an AI. Please continue your conversation with the user.<|im_end|>\n<|im_start|>user\nWrite a short python program demonstrating advanced features. Don't explain it.<|im_end|>\n<|im_start|>assistant" \
    -n 128 \
    --mpi-layer-split 0.1,0.15,0.15,0.15,0.15,0.15,0.15/1.0 \
    --ignore-eos \
    --temp -1.0 \
    --repeat-last-n 0 \
    --draft 3 \
    -c 1024 \
    -pa 0.1 \
    --numa \
    -pr 0.0 \
    -np 3 \
    -ps 0.9
popd
