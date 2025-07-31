1. Create a ReLoRA env using pip or conda. Ensure python version >= 3.10
2. Go to environemnet and execute pip install -r requirements.txt
3. pip install -e .
4. Install flash attention using:
TORCH_CUDA_ARCH_LIST="7.5" pip install flash_attn --no_cache_dir --no_build_isolation
5. Execute the pretokenizer:
--save_dir "hg19_512_1_2_ds"
--tokenizer "EleutherAI/gpt_neox_20b"
--train_file "hg19_512_1_2/train.csv"
--validation_file "hg19_512_1_2/val.csv"
--test_file "hg19_512_1_2/test.csv"
--num_cpu 8
--sequence_length 512
6. Execute the torch run main code
   torchrun --nproc_per_node 1 torchrun_main.py
   --model_config configs/llama_20m_50k.json
   --batch_size 8
   --total_batch_size 16
   --lr 1e-3
   --max_length 512
   --use_peft True
   --relora 3000
   --cycle_length 3000
   --restart_warmup_steps 100
   --schedular cosine_restarts
   --warm_steps 500
   --reset_optimizer_on_relora True
   --num_training_steps 21000
   --save_every 1000
   --eval_every 1000
   --dataset_path $DATA_PATH
   --warmed_up_model checkpoints/___/___
   
 
