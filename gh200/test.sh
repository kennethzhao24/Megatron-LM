apptainer exec --nv \
  --bind /u/yzhao25/Megatron-LM/examples/run_simple_mcore_train_loop.py:/root/Megatron-LM/examples/run_simple_mcore_train_loop.py \
  /u/yzhao25/slime_containers/slime-base.sif \
  torchrun --nproc_per_node=2 /root/Megatron-LM/examples/run_simple_mcore_train_loop.py
