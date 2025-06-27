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
