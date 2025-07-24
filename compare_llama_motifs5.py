import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu
import deBrujinAdjGraph as deBAG

file = open("nohup2.out","r")
positive_motifs = {}
negative_motifs = {}
positive_motifs_null = {}
negative_motifs_null = {}
positive_flag = 0
pos_freq_sums = 0
neg_freq_sums = 0
positive_flag_full_seq = 0
for line in file.readlines():
	if line.find("predicted_positive_vocab")!=-1:
		positive_flag = 1
		positive_flag_full_seq = -1
	if line.find("predicted_negative_vocab")!=-1:
		positive_flag = 0
		positive_flag_full_seq = -1
	if line.find("predicted_positive_vocab_full_seq")!=-1:
		positive_flag_full_seq = 1
		positive_flag = -1
	if line.find("predicted_negative_vocab_full_seq")!=-1:
		positive_flag_full_seq = 0
		positive_flag = -1
	if line.find(":")==8 and positive_flag==1:
		motif = line[:8]
		freq = int(line[9:])
		positive_motifs[motif]=freq
		pos_freq_sums+=freq
	if line.find(":")==8 and positive_flag==0:
		motif = line[:8]
		freq = int(line[9:])
		negative_motifs[motif]=freq
		neg_freq_sums+=freq
	if line.find(":")==8 and positive_flag_full_seq==1:
                motif = line[:8]
                freq = int(line[9:])
                positive_motifs_null[motif]=freq
	if line.find(":")==8 and positive_flag_full_seq==0:
		motif = line[:8]
		freq = int(line[9:])
		negative_motifs_null[motif]=freq

positive_motifs_values = []
positive_motifs_null_values= []

for motif in positive_motifs.keys():
	if motif in positive_motifs_null.keys() and positive_motifs[motif]>0 and positive_motifs_null[motif]>0:
		positive_motifs_values.append(positive_motifs[motif])
		positive_motifs_null_values.append(positive_motifs_null[motif])

negative_motifs_values = []
negative_motifs_null_values= []

for motif in negative_motifs.keys():
	if motif in negative_motifs_null.keys() and negative_motifs[motif]>0 and negative_motifs_null[motif]>0:
        	negative_motifs_values.append(negative_motifs[motif])
        	negative_motifs_null_values.append(negative_motifs_null[motif])

print("positive_motifs\n",len(positive_motifs_values))
print("negative_motifs\n",len(negative_motifs_values))

print("positive_motifs_null\n",len(positive_motifs_null_values))
print("negative_motifs_null\n",len(negative_motifs_null_values))


ratio = pos_freq_sums/neg_freq_sums
print("positive_to_negative_cumulative_frequency_ratio:",ratio)

U1, p = mannwhitneyu(np.array(positive_motifs_null_values)/np.max(np.array(positive_motifs_null_values)), np.array(positive_motifs_values)/np.max(np.array(positive_motifs_values)),\
alternative='two-sided', method="asymptotic")
print("p_value_positive:",p)
print("U:",U1)

U1, p = mannwhitneyu(np.array(negative_motifs_null_values)/np.max(np.array(negative_motifs_null_values)), np.array(negative_motifs_values)/np.max(np.array(negative_motifs_values)),\
alternative='two-sided', method="asymptotic")
print("p_value_negative:",p)
print("U:",U1)

positive_motifs_values = []
negative_motifs_values = []

for motif in positive_motifs.keys():
        if motif in negative_motifs.keys() and negative_motifs[motif]>0 and positive_motifs[motif]>0:
                positive_motifs_values.append(positive_motifs[motif])
                negative_motifs_values.append(negative_motifs[motif])

U1, p = mannwhitneyu(np.array(positive_motifs_values), np.array(negative_motifs_values),alternative='two-sided', method="asymptotic")
print("p_value_positive_vs_negative:",p)
print("U:",U1)

forty_percent_delta_motifs=[]
fifty_percent_delta_motifs=[]
sixty_percent_delta_motifs=[]
seventy_percent_delta_motifs = []
eighty_percent_delta_motifs = []

delta_pos_freq_norm = []

delta_neg_freq_norm = []

for key in positive_motifs.keys() :
	if key in negative_motifs.keys() and np.absolute(positive_motifs[key]-negative_motifs[key]*ratio)/max(positive_motifs[key],negative_motifs[key]*ratio)>0.4:
		forty_percent_delta_motifs.append(key)
	if key in negative_motifs.keys() and np.absolute(positive_motifs[key]-negative_motifs[key]*ratio)/max(positive_motifs[key],negative_motifs[key]*ratio)>0.5:
		fifty_percent_delta_motifs.append(key)
	if key in negative_motifs.keys() and np.absolute(positive_motifs[key]-negative_motifs[key]*ratio)/max(positive_motifs[key],negative_motifs[key]*ratio)>0.6:
		sixty_percent_delta_motifs.append(key)
	if key in negative_motifs.keys() and np.absolute(positive_motifs[key]-negative_motifs[key]*ratio)/max(positive_motifs[key],negative_motifs[key]*ratio)>0.7:
		seventy_percent_delta_motifs.append(key)
	if key in negative_motifs.keys() and np.absolute(positive_motifs[key]-negative_motifs[key]*ratio)/max(positive_motifs[key],negative_motifs[key]*ratio)>0.8:
		eighty_percent_delta_motifs.append(key)
		delta_pos_freq_norm.append(positive_motifs[key])
		delta_neg_freq_norm.append(int(negative_motifs[key]*ratio))

print("40_percent_delta_octamer_motifs",forty_percent_delta_motifs)
print("50_percent_octamer_motifs",fifty_percent_delta_motifs)
print("60_percent_octamer_motifs",sixty_percent_delta_motifs)
print("70_percent_octamer_motifs",seventy_percent_delta_motifs)
print("80_percent_octamer_motifs",eighty_percent_delta_motifs)

fig = plt.subplots(figsize =(40, 8))
x = np.arange(len(eighty_percent_delta_motifs))  # the label locations
width = 0.25  # the width of the bars
multiplier = 0
#plt.figure(1, figsize=(50,10))

x2 = [i + width for i in x] 
plt.bar(x, delta_pos_freq_norm, color='red',width=width,label="positive")
plt.bar(x2, delta_neg_freq_norm, color='blue',width=width,label="negative")
plt.xlabel('Motifs') 
plt.ylabel('Frequency')
plt.xticks(x,eighty_percent_delta_motifs)
#plt.twinx()
plt.legend()
plt.savefig("motif_differences2.png")

"""
delta_pos_freq_norm = []
delta_neg_freq_norm = []
delta_motifs = {}
for key in positive_motifs.keys() :
        if key in negative_motifs.keys():
                delta_motifs[key]=np.absolute(positive_motifs[key]-negative_motifs[key]*ratio)

sorted_delta_motif_diffs = np.argsort(delta_motifs.keys())
"""
num_predicted_positive = 0
num_predicted_negative = 0
predicted_positive_vocab = {}
predicted_negative_vocab = {}
predicted_positive_vocab_ar = {}
predicted_negative_vocab_ar = {}
predicted_positive_vocab_full_seq = {}
predicted_negative_vocab_full_seq = {}
predicted_positive_vocab_full_seq_ar = {}
predicted_negative_vocab_full_seq_ar = {}
octamer_permutations = deBAG.init_all_kmer_permutation_counts('ATCG', 8)
for permutation in list(octamer_permutations.keys()):
	predicted_positive_vocab[permutation] = 0
	predicted_positive_vocab_ar[permutation] = []
	predicted_positive_vocab_full_seq[permutation] = 0
	predicted_positive_vocab_full_seq_ar[permutation] = []
	predicted_negative_vocab[permutation] = 0
	predicted_negative_vocab_ar[permutation] = []
	predicted_negative_vocab_full_seq[permutation] = 0
	predicted_negative_vocab_full_seq_ar[permutation] = []
file2 = open("nohup2.out","r")
for i,line in enumerate(file2.readlines()):
	for k in range(3):
		if line.find("predicted_class_name LABEL_0")!=-1:
			predicted_class_name = "LABEL_0"
		elif line.find("predicted_class_name LABEL_1")!=-1:
			predicted_class_name = "LABEL_1"
		if line.find("sorted_attention_token"+str(k+1))!=-1:
			print("line",i)
			decoded_token = line[len("sorted_attention_token"+str(k+1)):]
			decoded_token_no_space = decoded_token.replace(" ","")
			decoded_token_no_space = decoded_token_no_space.replace("\n","")
			decoded_token_no_space = decoded_token_no_space.replace("<|endoftext|>","")
			decoded_token_no_space_octamers = deBAG.get_kmer_count_from_sequence(decoded_token_no_space, k=8, cyclic=False)
			set_of_others = list(set(list(octamer_permutations.keys())) - set(list(decoded_token_no_space_octamers.keys())))
			print("len(decoded_token_no_space_octamers)",len(decoded_token_no_space_octamers))
			print("len(set_of_others)",len(set_of_others))
			if predicted_class_name=="LABEL_0":
				for token in list(decoded_token_no_space_octamers.keys()):
					predicted_negative_vocab_ar[token].append(decoded_token_no_space_octamers[token])
				for token in set_of_others:
					predicted_negative_vocab_ar[token].append(0)
			else:
				for token in list(decoded_token_no_space_octamers.keys()):
					predicted_positive_vocab_ar[token].append(decoded_token_no_space_octamers[token])
				for token in set_of_others:
					predicted_positive_vocab_ar[token].append(0)
p_value_dict = {}
for permutation in list(octamer_permutations.keys()):
	print(np.count_nonzero(predicted_positive_vocab_ar[permutation]))
	print(np.count_nonzero(predicted_negative_vocab_ar[permutation]))
	print(len(predicted_positive_vocab_ar[permutation]))
	print(len(predicted_negative_vocab_ar[permutation]))
	U1, p = mannwhitneyu(np.array(predicted_positive_vocab_ar[permutation]), np.array(predicted_negative_vocab_ar[permutation]),alternative='two-sided', method="asymptotic")
	print("permutation",permutation)
	print("p_value_positive_vs_negative:",p)
	print("U:",U1)
	p_value_dict[permutation] = p
	
top_10_keys = sorted(p_value_dict, key=p_value_dict.get)[:10]
motifs = []
p_values = []
num_pos = []
num_neg = [] 
for key in top_10_keys:
	print("motif:",key)
	print("p_value:",p_value_dict[key])
	print("num_positive",np.count_nonzero(predicted_positive_vocab_ar[key]))
	print("num_negative",np.count_nonzero(predicted_negative_vocab_ar[key]))
	motifs.append(key)
	p_values.append(p_value_dict[key])
	num_pos.append(np.count_nonzero(predicted_positive_vocab_ar[key]))
	num_neg.append(np.count_nonzero(predicted_negative_vocab_ar[key]))
	
motif_df = pd.DataFrame(data=list(zip(motifs,p_values,num_pos,num_neg)),columns=["motif","p_value","num_pos","num_neg"])
motif_df.to_csv("motif_df.csv")

fig = plt.subplots(figsize =(40, 8))
x = np.arange(10)  # the label locations
width = 0.25  # the width of the bars
multiplier = 0
#plt.figure(1, figsize=(50,10))

x2 = [i + width for i in x] 
plt.bar(x, num_pos, color='red',width=width,label="positive")
plt.bar(x2, num_neg, color='blue',width=width,label="negative")
plt.xlabel('Motifs') 
plt.ylabel('Frequency')
plt.xticks(x,motifs)
#plt.twinx()
plt.legend()
plt.savefig("top_10_lowest_p_val_motifs.png")
