import numpy as np
import pandas as pd
import os
import math

genome = "hg19"
window="1024"
tss_motifs_chr1_chr2 = pd.read_csv("cage_combined_"+genome+"_chr1_chr2_"+window+"_context_window.csv")
tss_motifs_chr3_chr4 = pd.read_csv("cage_combined_"+genome+"_chr3_chr4_"+window+"_context_window.csv")
tss_motifs_chr5_chr6 = pd.read_csv("cage_combined_"+genome+"_chr5_chr6_"+window+"_context_window.csv")
tss_motifs_chr7_chr8 = pd.read_csv("cage_combined_"+genome+"_chr7_chr8_"+window+"_context_window.csv")
tss_motifs_chr9_chr10 = pd.read_csv("cage_combined_"+genome+"_chr9_chr10_"+window+"_context_window.csv")
tss_motifs_chr11_chr12 = pd.read_csv("cage_combined_"+genome+"_chr11_chr12_"+window+"_context_window.csv")
tss_motifs_chr13_chr14 = pd.read_csv("cage_combined_"+genome+"_chr13_chr14_"+window+"_context_window.csv")
tss_motifs_chr15_chr16 = pd.read_csv("cage_combined_"+genome+"_chr15_chr16_"+window+"_context_window.csv")
if genome=="hg19":
	tss_motifs_chr17_chr18 = pd.read_csv("cage_combined_"+genome+"_chr17_chr18_"+window+"_context_window.csv")
	tss_motifs_chr19_chr20 = pd.read_csv("cage_combined_"+genome+"_chr19_chr20_"+window+"_context_window.csv")
	tss_motifs_chr21_chr22 = pd.read_csv("cage_combined_"+genome+"_chr21_chr22_"+window+"_context_window.csv")
if genome=="mm9":
	tss_motifs_chr17_chr18_chr19 = pd.read_csv("cage_combined_"+genome+"_chr17_chr18_chr19_"+window+"_context_window.csv")	
tss_motifs_chrX_chrY = pd.read_csv("cage_combined_"+genome+"_chrX_chrY_"+window+"_context_window.csv")

tss_motifs_combined_chr1_to_chr12 = pd.concat([tss_motifs_chr1_chr2,tss_motifs_chr3_chr4,tss_motifs_chr5_chr6,tss_motifs_chr7_chr8,tss_motifs_chr9_chr10,
						tss_motifs_chr11_chr12],ignore_index=True)
if genome=="hg19":
	tss_motifs_combined_chr13_to_chrY = pd.concat([tss_motifs_chr13_chr14,tss_motifs_chr15_chr16,tss_motifs_chr17_chr18,
	                                               tss_motifs_chr19_chr20,tss_motifs_chr21_chr22,tss_motifs_chrX_chrY],ignore_index=True)
if genome=="mm9":
	tss_motifs_combined_chr13_to_chrY = pd.concat([tss_motifs_chr13_chr14,tss_motifs_chr15_chr16,
						      tss_motifs_chr17_chr18_chr19,tss_motifs_chrX_chrY],ignore_index=True)

tss_motifs_combined = pd.concat([tss_motifs_combined_chr1_to_chr12,tss_motifs_combined_chr13_to_chrY],ignore_index=True)
tss_motifs_combined2 = tss_motifs_combined.sample(frac=1.0)
tss_motifs_combined2.reset_index(drop=True,inplace=True)
print(tss_motifs_combined2)
tss_motifs_combined2.to_csv("cage_combined_"+genome+"_chromosomes_combined_"+window+"_context_window.csv",index=False)
