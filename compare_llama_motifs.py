import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

file = open("nohup.out","r")
positive_motifs = {}
negative_motifs = {}
positive_flag = 0
pos_freq_sums = 0
neg_freq_sums = 0
for line in file.readlines():
	if line.find("predicted_positive_vocab")!=-1:
		positive_flag = 1
	elif line.find("predicted_negative_vocab")!=-1:
		positive_flag = 0
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

print("positive_motifs\n",positive_motifs)
print("negative_motifs\n",negative_motifs)

ratio = pos_freq_sums/neg_freq_sums
print("positive_to_negative_cumulative_frequency_ratio:",ratio)

ten_delta_motifs=[]
twenty_delta_motifs=[]
thirty_delta_motifs=[]

ten_delta_pos_freq_norm = []

ten_delta_neg_freq_norm = []

for key in positive_motifs.keys():
	if np.absolute(positive_motifs[key]-negative_motifs[key]*ratio)>10:
		ten_delta_motifs.append(key)
		ten_delta_pos_freq_norm.append(positive_motifs[key])
		ten_delta_neg_freq_norm.append(int(negative_motifs[key]*ratio))
	if np.absolute(positive_motifs[key]-negative_motifs[key]*ratio)>20:
		twenty_delta_motifs.append(key)
	if np.absolute(positive_motifs[key]-negative_motifs[key]*ratio)>30:
		thirty_delta_motifs.append(key)

print("10_delta_octamer_motifs",ten_delta_motifs)
print("20_delta_octamer_motifs",twenty_delta_motifs)
print("30_delta_octamer_motifs",thirty_delta_motifs)

fig = plt.subplots(figsize =(256, 8))
x = np.arange(len(ten_delta_motifs))  # the label locations
width = 0.25  # the width of the bars
multiplier = 0
#plt.figure(1, figsize=(50,10))

x2 = [i + width for i in x] 
plt.bar(x, ten_delta_pos_freq_norm, color='red',width=width,label="positive")
plt.bar(x2, ten_delta_neg_freq_norm, color='blue',width=width,label="negative")
plt.xlabel('Motifs') 
plt.ylabel('Frequency')
plt.xticks(x,ten_delta_motifs)
#plt.twinx()
plt.legend()
plt.savefig("motif_differences.png")
