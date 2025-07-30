import numpy as np
import os
import pandas as pd
import sys
from sklearn.model_selection import train_test_split

arg1 = str(sys.argv[1])
arg2 = str(sys.argv[2])
df1 = pd.read_csv("cage_combined_hg19_chromosomes_combined_1024_context_window.csv")

"""
sentences = []
labels = []
chroms = []
for ind in range(len(df1.index)):
	sequence = df1["fasta_seq"].loc[ind]
	label = df1["cage_tag"].loc[ind]
	words = [''.join(sequence[i:i+6]+' ') for i in range(0,len(sequence)-5,1)]
	sentence = ''.join(words)
	sentence = sentence.strip(' ')
	sentences.append(sentence)
	labels.append(label)
	chroms.append(df1["chrom"].loc[ind])

df2 = pd.DataFrame(data=list(zip(sentences,labels,chroms)),columns=["fasta_seq","cage_tag","chrom"])
df2.to_csv("cage_combined_hg19_chromosomes_combined_100_context_window_"+arg1+"_mer.csv",index=False)
"""
df2 = df1
#train, dev = train_test_split(df2, test_size=0.33, random_state=99)
dev = df2[df2["chrom"]=="chr"+arg1]
test = df2[df2["chrom"]=="chr"+arg2]
rem1 = df2[df2["chrom"]!="chr"+arg1]
train = rem1[rem1["chrom"]!="chr"+arg2]

train2 = pd.DataFrame()
train2["sequence"] = train["fasta_seq"].to_list()
train2["label"] = train["cage_tag"].to_list()

dev2 = pd.DataFrame()
dev2["sequence"] = dev["fasta_seq"].to_list()
dev2["label"] = dev["cage_tag"].to_list()

test2 = pd.DataFrame()
test2["sequence"] = test["fasta_seq"].to_list()
test2["label"] = test["cage_tag"].to_list()

dir = "hg19_1024_"+arg1+"_"+arg2
os.mkdir(dir)
train2.to_csv(os.path.join(dir,"train.csv"),index=False)
dev2.to_csv(os.path.join(dir,"dev.csv"),index=False)
test2.to_csv(os.path.join(dir,"test.csv"),index=False)
