### Please cite our paper 
"A Transformer based method for the Cap Analysis of 1 Gene Expression and Gene Expression Tag associated 2 5’ cap site prediction in RNA 3"
https://doi.org/10.1080/15476286.2026.2629530
Authors: \
Dibya Kanti Haldar \
Centre for Computational and Data Sciences, Indian Institute of Technology Kharagpur, West
Bengal, India-721302 \
Avik Pramanick \
Department of Computer Science and Engineering, Indian Institute of Technology Kharagpur,
West Bengal, India-721302 \
Chandrama Mukherjee \
Institute of Health Sciences, Presidency University, Kolkata, West Bengal, India - 700073 \
Pralay Mitra (Corresponding Author) \
Department of Computer Science and Engineering, Indian Institute of Technology Kharagpur.
### Third-Party Code Acknowledgments

This repository includes modified components from:

- [Hugging Face Transformers](https://github.com/huggingface/transformers)  
  Licensed under the Apache License 2.0.

- [ReLoRA](https://github.com/Guitaricet/relora)  
  Licensed under the Apache License 2.0.
All modifications are documented in the code and this repository fully complies with the original licenses.
### Here are the steps for reproducing our results.
1. 	Download and extract the fantom5 CAGE peak bed files from this path: "https://fantom.gsc.riken.jp/5/datafiles/latest
extra/CAGE_peaks/hg19.cage_peak_phase1and2combined_coord.bed.gz"
2. 	Download and extract hg19.fa from "https://hgdownload.cse.ucsc.edu/goldenPath/hg19/bigZips/hg19.fa.gz"
3. 	Convert the hg19 genome assembly’s CAGE peak bed file to a csv file with the name hg19_cage_peak_phase1and2combined_coord.csv
4. 	Create the capping sequences for use in later steps:
   	python3 preprocess_and_create_capping_dataset.py [CONTEXT_WINDOW] chr[1] chr[2]  \
   	where [CONTEXT_WINDOW] stands for the length of the sequence fragment on which prediction would be made, [1] stands for chromosome number c, and [2] stands \
   	for chromosome number c+1. \
   	Input: \
   	"hg19_cage_peak_phase1and2combined_coord.csv" \
   	Output: \
   	"cage_combined_hg19_chr[1]_chr[2]_[CONTEXT_WINDOW]_context_window.csv" \
   	Execute step 1 in a loop to obtain data for all chromosomes. 
5. 	Combine the chromosome wise capping sequences in one csv file. Change value of window (line #7) from 1024 to 512:
   	python3 combine_chromosome_wise_datasets.py \
   	Inputs: \
   	"cage_combined_hg19_chr1_chr2_[CONTEXT_WINDOW]_context_window.csv" \
   	.... \
   	"cage_combined_hg19_chrX_chrY_[CONTEXT_WINDOW]_context_window.csv" \
   	Output: \
   	"cage_combined_hg19_chromosomes_combined _[CONTEXT_WINDOW]_context_window.csv" 
6. 	Split the dataset into train, dev and test for the pretokenization step. \
   	python3 train_test_divide.py [1] [2] [CONTEXT_WINDOW] where \
   	[1] stands for the validation set chromosome, [2] stands for the test set chromosome, and [CONTEXT_WINDOW] stands for the length of the input sequence. \
   	Input: \
   	"cage_combined_hg19_chromosomes_combined_[CONTEXT_WINDOW]_context_window.csv" \
   	Outputs:
   	"hg19_[CONTEXT_WINDOW]_[1]_[2]/train.csv" \
   	"hg19_[CONTEXT_WINDOW]_[1]_[2]/dev.csv" \
   	"hg19_[CONTEXT_WINDOW]_[1]_[2]/test.csv" 
7. 	Create a ReLoRA env using pip or conda. Ensure python version >= 3.10
8. 	Go to environemnet and execute pip install -r requirements.txt
9. 	pip install -e .
10. Install flash attention using:
TORCH_CUDA_ARCH_LIST="7.5" pip install flash_attn --no_cache_dir --no_build_isolation
11. Execute the pretokenizer:
    python3 pretokenize.py \
   --save_dir "hg19_[CONTEXT_WINDOW]_[1]_[2]_ds" \
   --tokenizer "EleutherAI/gpt_neox_20b" \
   --train_file "hg19_[CONTEXT_WINDOW]_[1]_[2]/train.csv" \
   --validation_file "hg19_[CONTEXT_WINDOW]_[1]_[2]/dev.csv" \
   --test_file "hg19_[CONTEXT_WINDOW]_[1]_[2]/test.csv" \
   --num_cpu 8 \
   --sequence_length [CONTEXT_WINDOW] \
   where [1], [2] and [CONTEXT_WINDOW] hold the same meaning as step 3.
   The tokenized files are stored in "hg19_[CONTEXT_WINDOW]_[1]_[2]_ds/None_EleutherAI_gpt-neox-20b_[CONTEXT_WINDOW]"    
12. Perform the llama warmup steps:
   mkdir checkpoints
   export DATA_PATH="hg19_[CONTEXT_WINDOW]_[1]_[2]_ds/None_EleutherAI_gpt-neox-20b_[CONTEXT_WINDOW]"
   torchrun --nproc-per-node 1 torchrun_main.py \
       --model_config configs/llama_20m_50K.json \
       --dataset_path $DATA_PATH \
       --batch_size 8 \
       --total_batch_size 16 \
       --lr 5e-4 \
       --max_length [CONTEXT_WINDOW] \
       --save_every 1000 \
       --eval_every 1000 \
       --num_training_steps 1000 \
       --tags warm_start_20M_50K \
    The warm up model will be saved in the checkpoint directory. \
    Example path of saved warm up model checkpoint: "checkpoints/light-feather-157/model_1000"
13. Execute the torch run main code for pretraining using llama + relora
    export CHECKPOINT_PATH="checkpoints/light-feather-157/model_1000"
   	torchrun --nproc_per_node 1 torchrun_main.py \
   	--model_config configs/llama_20m_50k.json \
   	--batch_size 16 \
   	--total_batch_size 32 \
   	--lr 1e-3 \
   	--max_length [CONTEXT_WINDOW] \
   	--use_peft True \
   	--relora 4000 \
   	--cycle_length 4000 \
   	--restart_warmup_steps 100 \
   	--schedular cosine_restarts \
   	--warm_steps 500 \
   	--reset_optimizer_on_relora True \
   	--num_training_steps 21000 \
   	--save_every 1000 \
   	--eval_every 1000 \
   	--dataset_path $DATA_PATH \
   	--warmed_up_model $CHECKPOINT_PATH \
    Example path of saved pretrained model using llama + relora: "checkpoints/classic-waterfall-158/model_14197" \
    Here [CONTEXT_WINDOW] holds the same meaning as step 3.
14. Execute the finetuning using llama + lora.
    Later versions of the transformer library would require you to comment the line having send_example_telemetry() (line# 247).
    Set tokenizer=None when initilializing Trainer (line# 612) if tokenization already performed in the earlier steps:
    python3 run_glue.py hg19_[CONTEXT_WINDOW]_[1]_[2].json
    The json file "hg19_[CONTEXT_WINDOW]_[1]_[2].json" can be defined as follows: \
    { \
		"model_name_or_path":"checkpoints/classic-waterfall-158/model_14197", \
    	"tokenizer_name": "EleutherAI/gpt-neox-20b", \
		"use_fast_tokenizer":true, \
		"model_revision":"main", \
		"ignore_mismatched_sizes":false, \
    	"max_seq_length": [CONTEXT_WINDOW], \
    	"overwrite_cache":false, \
    	"pad_to_max_length":true, \
    	"dataset_name":"hg19_[CONTEXT_WINDOW]_[1]_[2]_ds/None_EleutherAI_gpt-neox-20b_[CONTEXT_WINDOW]", \
    	"do_train":true, \
    	"do_predict":false, \
    	"do_eval":true, \
		"eval_strategy":"steps", \
		"max_train_samples":300000, \
		"max_eval_samples":20000, \
		"num_train_epochs":5, \
		"eval_steps":1000, \
		"per_gpu_train_batch_size":32 \
 	}  \
	Here [1], [2] and [CONTEXT_WINDOW] hold the same meaning as step 3. \
	The finetuned checkpoints will get saved in a folder named "trainer_output" \
	You can change the checkpoint folder name to something like "hg19_[CONTEXT_WINDOW]_[1]_[2]_output"
15. Get the prediction and the attention weights:
    Add this ‘self.ref_token_id = 2’ in transformer_interpreter library located at ‘lib/python3.12/site-packages
	transformers_interpret/explainer.py’ line 125 since much of the implementation of this library is incomplete.
   	nohup python3 run_glue_predict_with_postprocessing.py ft_data.json & \
    The json file ft_data.json can be something like: \
    { \
		"model_name_or_path":"hg19_[CONTEXT_WINDOW]_[1]_[2]_output/trainer_output", \
    	"tokenizer_name": "EleutherAI/gpt-neox-20b", \
		"use_fast_tokenizer":true, \
		"model_revision":"main", \
		"ignore_mismatched_sizes":false, \
    	"max_seq_length":[CONTEXT_WINDOW], \
    	"overwrite_cache":false, \
    	"pad_to_max_length":true, \
    	"dataset_name":"hg19_[CONTEXT_WINDOW]_[1]_[2]_ds/None_EleutherAI_gpt-neox-20b_512", \
    	"do_train":false, \
    	"do_predict":true, \
    	"do_eval":true, \
		"eval_strategy":"steps", \
		"max_train_samples":300000, \
		"max_eval_samples":2000, \
		"max_predict_samples":5000, \
		"num_train_epochs":5, \
		"eval_steps":1000, \
		"per_gpu_train_batch_size":16 \
	} \
	Here [1], [2] and [CONTEXT_WINDOW] hold the same meaning as step 3. \
	Outputs: \
	"nohup.out" \
	"octamer_sequence_df_10_dist.csv"
16. To get the predicted motifs in the ascending order of p-value:
    python3 compare_llama_predicted_motifs.py
   	Inputs:
   	"nohup.out" \
   	"octamer_sequence_df_10_dist.csv" \
   	Outputs: \
   	"motif_differences2.png" \
   	"motif_df.csv" \
   	"motif_df2_10_dist.csv" [Alternatively, you can use a more strict 5_dist] \
   	"motifs_with_p_vals_and_seqs_10_dist.fa" \
   	"motifs_with_p_vals_and_seqs_10_dist_[octamer_sequence_motif].fa" \
   	"lowest_p_val_motifs_10_dist.png" \
   	Here, [octamer_sequence_motif] refers to any predicted octamer motif with positive vs negative p_value less than 0.01.
17.	Install fimo using the installation instructions in this page: https://meme-suite.org/meme/doc/install.html?man_type=web.
	Download the JASPAR meme files from here: https://jaspar2022.genereg.net/downloads/ \
    Execute the following code to get the TF motifs matching the sequnece motifs:  \
    python3 main_fimo.py [octamer_sequence_motif] \
    Inputs: \
    [Path to the folder containing .meme files] \
    "motifs_with_p_vals_and_seqs_10_dist_[octamer_sequence_motif].fa" \
    Output: \
    "fimo_results_motifs_30_[octamer_sequence_motif]" \
    Here, the output folder contains all the motif matches.
If you face in issues with the code, data or installation, you are free to contact us at email-id:"dibyohaldar72@gmail.com".
