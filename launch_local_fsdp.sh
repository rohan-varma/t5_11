torchrun --nnodes=1 --nproc_per_node=8 --rdzv_id=100 --rdzv_endpoint="localhost:5678" trainer.py  --mode=fsdp --model=GPT13B --init_only True
