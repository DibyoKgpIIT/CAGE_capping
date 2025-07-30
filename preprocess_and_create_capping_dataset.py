# importing the libraries
import numpy as np
import pandas as pd
import os
import math
from pyfaidx import Fasta
import random
import sys

# chromosome mapping for the mouse genome
genome = Fasta('hg19.fa')
#genome = Fasta('mm9.fa')
# obtain the chromosome names
chr1 = "chr1"
chr2 = "chr2"
chr3 = ""
CONTEXT = 256
if len(sys.argv)>=2:
	CONTEXT = int(sys.argv[1])
if len(sys.argv)>=3:
	chr1 = sys.argv[2]
if len(sys.argv)>=4:
	chr2 = sys.argv[3]
if len(sys.argv)>=5:
        chr3 = sys.argv[4]

# obtain cage sites and fasta
cage_sites_and_fasta = pd.read_csv("hg19_cage_peak_phase1and2combined_coord.csv")
# cage_sites_and_fasta = pd.read_csv("fasta_chr_all_CAGE_plus_plus_mm9.csv")
# obtain the nucleotide sequences in the vicinity of the cage tags
cage_peaks = []
chr_nums = []
cage_start_sites = []
cage_end_sites = []
seq_lengths = []
seq_cage_begins = []
seq_cage_ends = []
near_cage_starts = []
near_cage_ends = []
sequences = []
for ind in range(len(cage_sites_and_fasta.index)):
    chr_num = cage_sites_and_fasta["chromosomeID"].loc[ind]
    cage_start_site = cage_sites_and_fasta["start_coordinate"].loc[ind]
    cage_end_site = cage_sites_and_fasta["end_coordinate"].loc[ind]
    cage_peak = str(chr_num)+":"+str(cage_start_site)+"-"+str(cage_end_site)
    cage_begin = CONTEXT//2
    cage_end = CONTEXT//2+int(cage_end_site)-int(cage_start_site)
    near_cage_start = chr_num+":"+str(int(cage_start_site)//CONTEXT *CONTEXT)
    if(chr_num==chr1 or chr_num==chr2 or chr_num==chr3):
        try:
            chrom_seq = genome[chr_num][int(cage_start_site)-CONTEXT//2:int(cage_start_site)+CONTEXT//2].seq.upper()
            #chrom_seq = genome[chr_num][int(cage_start_site)-500:int(cage_start_site)+500].seq.upper()
            seq_len = len(chrom_seq)
            if seq_len!=CONTEXT or chrom_seq.find('N')!=-1:
                continue
            else:
                cage_peaks.append(cage_peak)
                chr_nums.append(chr_num)
                cage_start_sites.append(int(cage_start_site))
                cage_end_sites.append(int(cage_end_site))
                seq_cage_begins.append(int(cage_begin))
                seq_cage_ends.append(int(cage_end))
                near_cage_starts.append(near_cage_start)
                sequences.append(chrom_seq)
                seq_lengths.append(int(seq_len))
        except:
            print("Error at:"+chr_num+":"+str(int(cage_start_site)-CONTEXT//2)+":"+str(int(cage_end_site)+CONTEXT//2))

cage_sites_and_fasta2p = pd.DataFrame(data = list(zip(cage_peaks,chr_nums,cage_start_sites,cage_end_sites,seq_lengths,seq_cage_begins,seq_cage_ends,near_cage_starts)),
					columns = ["cage_peak","chrom","cage_start","cage_end","seq_len","seq_begin","seq_end","near_cage_start"])


cage_sites_and_fasta3p = pd.DataFrame(data = list(zip(cage_peaks,sequences,chr_nums,cage_start_sites,cage_end_sites,seq_lengths,seq_cage_begins,seq_cage_ends,near_cage_starts)),
                                        columns = ["cage_peak","fasta_seq","chrom","cage_start","cage_end","seq_len","seq_begin","seq_end","near_cage_start"])
cage_sites_and_fasta_short_p = pd.DataFrame(data = list(zip(seq_cage_begins,seq_cage_ends,near_cage_starts)),
                                        columns = ["seq_begin","seq_end","near_cage_start"])
cage_sites_and_fasta_p = cage_sites_and_fasta3p.merge(cage_sites_and_fasta_short_p)
cage_sites_and_fasta_p["cage_tag"] = [1]*len(cage_sites_and_fasta_p.index)
print("cage sites and their neighbourhood nuclotide sequence:")
print(cage_sites_and_fasta_p.columns)
print(cage_sites_and_fasta_p)

# obtain nucleotide sequences away from cage tags

cage_peaks = []
chr_nums = []
cage_start_sites = []
cage_end_sites = []
seq_lengths = []
seq_cage_begins = []
seq_cage_ends = []
near_cage_starts = []
near_cage_ends = []
sequences = []
for ind in range(len(cage_sites_and_fasta.index)):
    chr_num = cage_sites_and_fasta["chromosomeID"].loc[ind]
    cage_start_site = cage_sites_and_fasta["start_coordinate"].loc[ind]
    cage_end_site = cage_sites_and_fasta["end_coordinate"].loc[ind]
    cage_peak = str(chr_num)+":"+str(cage_start_site)+"-"+str(cage_end_site)
    for j in range(2):
        upstream_dist = np.random.randint(10000,20000)
        cage_begin = int(cage_start_site)-upstream_dist
        cage_end = int(cage_end_site)-upstream_dist
        near_cage_start = chr_num+":"+str(int(cage_begin)//CONTEXT *CONTEXT)
        near_cage_end = chr_num+":"+str(int(cage_end)//CONTEXT *CONTEXT)
        if(chr_num==chr1 or chr_num==chr2 or chr_num==chr3):
            chrom_seq = genome[chr_num][int(cage_begin)//CONTEXT *CONTEXT:int(cage_begin)//CONTEXT *CONTEXT+CONTEXT].seq.upper()
            #chrom_seq = genome[chr_num][int(cage_begin)//1000 *1000:int(cage_begin)//1000 *1000+1000].seq.upper()
            seq_len = len(chrom_seq)
            print(seq_len,chrom_seq)
            if seq_len!=CONTEXT or chrom_seq.find('N')!=-1:
                continue
            else:
                cage_peaks.append(cage_peak)
                chr_nums.append(chr_num)
                cage_start_sites.append(int(cage_start_site))
                cage_end_sites.append(int(cage_end_site))
                seq_cage_begins.append(cage_begin-int(cage_begin)//CONTEXT *CONTEXT+upstream_dist)
                seq_cage_ends.append(cage_end-int(cage_begin)//CONTEXT *CONTEXT+upstream_dist)
                near_cage_starts.append(near_cage_start)
                near_cage_ends.append(near_cage_end)
                seq_lengths.append(int(seq_len))
                sequences.append(chrom_seq)
cage_sites_and_fasta2n = pd.DataFrame(data = list(zip(cage_peaks,chr_nums,cage_start_sites,cage_end_sites,seq_lengths,seq_cage_begins,seq_cage_ends,near_cage_starts)),
                                        columns = ["cage_peak","chrom","cage_start","cage_end","seq_len","seq_begin","seq_end","near_cage_start"])

print("cage_sites_and_fasta2n\n",cage_sites_and_fasta2n.columns)
print(cage_sites_and_fasta2n)
cage_sites_and_fasta3n = pd.DataFrame(data = list(zip(cage_peaks,sequences,chr_nums,cage_start_sites,cage_end_sites,seq_lengths,seq_cage_begins,seq_cage_ends,near_cage_starts)),
                                        columns = ["cage_peak","fasta_seq","chrom","cage_start","cage_end","seq_len","seq_begin","seq_end","near_cage_start"])
cage_sites_and_fasta_short_n = pd.DataFrame(data = list(zip(seq_cage_begins,seq_cage_ends,near_cage_starts,near_cage_ends)),
                                        columns = ["seq_begin","seq_end","near_cage_start","near_cage_end"])
print("len(cage_sites_and_fasta_short_n)",len(cage_sites_and_fasta_short_n))
# remove rows having start locations as in the positive cage tag dataset
cage_sites_and_fasta_short_common1 = cage_sites_and_fasta_short_p.merge(cage_sites_and_fasta_short_n,how="inner",on="near_cage_start")
cage_sites_and_fasta_short_common2 = cage_sites_and_fasta_short_p.merge(cage_sites_and_fasta_short_n,how="inner",left_on="near_cage_start",right_on="near_cage_end")
cage_sites_and_fasta_short_common1_data = list(zip(cage_sites_and_fasta_short_common1.seq_begin_y,
					cage_sites_and_fasta_short_common1.seq_end_y,
					cage_sites_and_fasta_short_common1.near_cage_start,
					cage_sites_and_fasta_short_common1.near_cage_end))
cage_sites_and_fasta_short_common1_cols = ["seq_begin","seq_end","near_cage_start","near_cage_end"]
cage_sites_and_fasta_short_common1_df = pd.DataFrame(data = cage_sites_and_fasta_short_common1_data, columns = cage_sites_and_fasta_short_common1_cols)
print("cage_sites_and_fasta_short_common1_df\n",cage_sites_and_fasta_short_common1_df)
cage_sites_and_fasta_short_common2_data = list(zip(cage_sites_and_fasta_short_common2.seq_begin_y,
					cage_sites_and_fasta_short_common2.seq_end_y,
					cage_sites_and_fasta_short_common2.near_cage_start_y,
					cage_sites_and_fasta_short_common2.near_cage_end))
cage_sites_and_fasta_short_common2_cols = ["seq_begin","seq_end","near_cage_start","near_cage_end"]
cage_sites_and_fasta_short_common2_df = pd.DataFrame(data = cage_sites_and_fasta_short_common2_data, columns = cage_sites_and_fasta_short_common2_cols)
print("cage_sites_and_fasta_short_common2_df",cage_sites_and_fasta_short_common2_df)
cage_sites_and_fasta_short_common = pd.concat([cage_sites_and_fasta_short_common1_df,cage_sites_and_fasta_short_common2_df])
cage_sites_and_fasta_short_common.drop_duplicates(inplace=True,keep="first",ignore_index=True)
print("len(cage_sites_and_fasta_short_common)",len(cage_sites_and_fasta_short_common))
cage_sites_and_fasta_short_n_uncommon = pd.concat([cage_sites_and_fasta_short_n,cage_sites_and_fasta_short_common])
cage_sites_and_fasta_short_n_uncommon.drop_duplicates(inplace=True,keep=False,ignore_index=True)
print("len(cage_sites_and_fasta_short_n_uncommon)",len(cage_sites_and_fasta_short_n_uncommon))
cage_sites_and_fasta_n = cage_sites_and_fasta3n.merge(cage_sites_and_fasta_short_n_uncommon)
cage_sites_and_fasta_n["cage_tag"] = [0]*len(cage_sites_and_fasta_n.index)
print("non cage sites and their neighbourhood nuclotide sequence:")
cage_sites_and_fasta_n = cage_sites_and_fasta_n.sample(frac=1.0).reset_index(drop=True).loc[:len(cage_sites_and_fasta_p)]
print(cage_sites_and_fasta_n.columns)
print(cage_sites_and_fasta_n)
 
## Combining the positive and negative datasets
cage_combined = pd.concat([cage_sites_and_fasta_p,cage_sites_and_fasta_n])
cage_combined.drop_duplicates(inplace=True,keep="first",ignore_index=True)
cage_combined.to_csv("cage_combined_mm9_"+chr1+"_"+chr2+"_"+str(CONTEXT)+"_context_window_all_cols.csv",index=False)
cage_combined_essential = cage_combined.drop(columns=["cage_peak","cage_start","cage_end","seq_begin",
						       "seq_end","near_cage_start","near_cage_end"],inplace=False)
cage_chr1_chr2 = cage_combined_essential.sample(frac=1)
cage_chr1_chr2.reset_index(drop=True,inplace=True)
print("Final dataframe with positive and negative examples:")
print(cage_chr1_chr2.columns)
print(cage_chr1_chr2)
if len(sys.argv)==4:
	cage_chr1_chr2.to_csv("cage_combined_hg19_"+chr1+"_"+chr2+"_"+str(CONTEXT)+"_context_window.csv",index=False)
elif len(sys.argv)==5:
	cage_chr1_chr2.to_csv("cage_combined_hg19_"+chr1+"_"+chr2+"_"+chr3+"_"+str(CONTEXT)+"_context_window.csv",index=False)
	
