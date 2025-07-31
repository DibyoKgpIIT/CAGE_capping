1. Create the capping sequences for use in later steps:
python3 preprocess_and_create_capping_dataset.py 1024 [1] [2] 
where [1] stands for chromosome number c, and [2] stands for chromosome number c+1.
Execute step 1 in a loop to obtain data for all chromosomes.
2. Combine the chromosome wise capping sequences in one csv file:
python3 combine_chromosome_wise_datasets.py
3. Split the dataset into train, dev and test for the pretokenization step. 
4. Create a ReLoRA env using pip or conda. Ensure python version >= 3.10
5. Go to environemnet and execute pip install -r requirements.txt
6. pip install -e .
7. Install flash attention using:
TORCH_CUDA_ARCH_LIST="7.5" pip install flash_attn --no_cache_dir --no_build_isolation
8. Execute the pretokenizer:
--save_dir "hg19_1024_1_2_ds"
--tokenizer "EleutherAI/gpt_neox_20b"
--train_file "hg19_1024_1_2/train.csv"
--validation_file "hg19_1024_1_2/dev.csv"
--test_file "hg19_1024_1_2/test.csv"
--num_cpu 8
--sequence_length 1024
9. Perform the llama warmup steps:
   export DATA_PATH=<path_to_preprocessed_data>
   torchrun --nproc-per-node 1 torchrun_main.py \
       --model_config configs/llama_20m_50k.json \
       --dataset_path $DATA_PATH \
       --batch_size 8 \
       --total_batch_size 16 \
       --lr 5e-4 \
       --max_length 1024 \
       --save_every 1000 \
       --eval_every 1000 \
       --num_training_steps 1000
       --tags warm_start_20M_50K
10. Execute the torch run main code for pretraining using llama + relora
   torchrun --nproc_per_node 1 torchrun_main.py
   --model_config configs/llama_20m_50k.json
   --batch_size 16
   --total_batch_size 32
   --lr 1e-3
   --max_length 1024
   --use_peft True
   --relora 4000
   --cycle_length 4000
   --restart_warmup_steps 100
   --schedular cosine_restarts
   --warm_steps 500
   --reset_optimizer_on_relora True
   --num_training_steps 21000
   --save_every 1000
   --eval_every 1000
   --dataset_path $DATA_PATH
   --warmed_up_model checkpoints/___/___
11. Execute the finetuning using llama + lora:
   python3 run_glue.py hg19_1024_1_2.json
   The json file can be defined as follows:
   {
		"model_name_or_path":"checkpoints/classic-waterfall-158/model_14197",
      "tokenizer_name": "EleutherAI/gpt-neox-20b",
		"use_fast_tokenizer":true,
		"model_revision":"main",
		"ignore_mismatched_sizes":false,
      "max_seq_length":1024,
      "overwrite_cache":false,
      "pad_to_max_length":true,
      "dataset_name":"hg19_1024_1_2_ds/None_EleutherAI_gpt-neox-20b_1024",
      "do_train":true,
      "do_predict":false,
      "do_eval":true,
		"eval_strategy":"steps",
		"max_train_samples":300000,
		"max_eval_samples":20000,
		"num_train_epochs":5,
		"eval_steps":1000,
		"per_gpu_train_batch_size":32
} 
12. Get the prediction and the attention weights:
   python3 run_glue_predict_with_postprocessing.py ft_data.json
13. To get the predicted motifs in the ascending order of p-value:
   python3 compare_llama_predicted_motifs.py
14.Install fimo using the installation instructions in this page: https://meme-suite.org/meme/doc/install.html?man_type=web.
    Execute the following code to get the TF motifs matching the sequnece motifs:  
    python3 main_fimo.py [sequence_motif_octamer]
