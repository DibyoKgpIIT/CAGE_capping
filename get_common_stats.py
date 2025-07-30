import os
import re
import numpy as np
import pandas as pd
from pyfaidx import Fasta

genome = Fasta('hg19.fa')

# Step 1: Define regex patterns for the motifs
# TATA box + UPE motif (TATAWADR + SSRCGCC)
tata_regex = re.compile(r'TATA[AT][AG]A[AG]')
upe_regex = re.compile(r'[CG][CG][AG]CGCC')

# Inr + DPE motif (YYANWYY + RGWYV)
inr_regex = re.compile(r'[CT][CT]A[AT][CT][CT]')
dpe_regex = re.compile(r'[AG][AG][AT][CT][ACG]')


def search_motifs(chrom, sequence, motif_name):
    """
    Search for TATA+UPE or Inr+DPE motifs in the provided sequence and extract sequences around the match.
    :param chrom: Chromosome name (e.g., 'chr1').
    :param sequence: The DNA sequence to search within.
    :param motif_name: Name of the ,motif to search
    :return: List of dictionaries with motif type, start, end, and extracted sequence.
    """
    results = []
    seq_len = len(sequence)
    if motif_name == "TATA":
        # Search for TATA
        for match in tata_regex.finditer(sequence):
            tata_start = match.start()
            tata_end = match.end()
            motif_sequence = sequence[tata_start:tata_end]
            results.append({
            'motif': motif_name,
            'chrom': chrom,
            'start': tata_start,
            'end': tata_end,
            'sequence': motif_sequence})

    # Search for UPE
    if motif_name == "UPE": 
        for match in upe_regex.finditer(sequence):
            upe_start = match.start()
            upe_end = match.end()
            motif_sequence = sequence[upe_start:upe_end]
            results.append({
            'motif': motif_name,
            'chrom': chrom,
            'start': upe_start,
            'end': upe_end,
            'sequence': motif_sequence
            })

    
    # Search for Inr
    if motif_name == "Inr":
        for match in inr_regex.finditer(sequence):
            inr_start = match.start()
            inr_end = match.end()
            motif_sequence = sequence[inr_start:inr_end]
            results.append({
            'motif': motif_name,
            'chrom': chrom,
            'start': inr_start,
            'end': inr_end,
            'sequence': motif_sequence
            })

    # Search for DPE
    if motif_name == "DPE": 
        for match in dpe_regex.finditer(sequence):
            dpe_start = match.start()
            dpe_end = match.end()
            motif_sequence = sequence[dpe_start:dpe_end]
            results.append({
            'motif': motif_name,
            'chrom': chrom,
            'start': dpe_start,
            'end': dpe_end,
            'sequence': motif_sequence
            })

    return results


cage_features = pd.read_csv("cage_combined_hg19_chromosomes_combined_1000_context_window.csv")
chromes = ["chr"+str(i) for i in range(1,23)]
chromes += ["chrX","chrY"]
print(chromes)
chr_count_pos = {}
chr_count_neg = {}

for chrom in chromes:
	chr_count_pos[chrom]=0
	chr_count_neg[chrom]=0

for ind in range(len(cage_features.index)):
	chr_num = cage_features["chrom"].loc[ind]
	label = cage_features["cage_tag"].loc[ind]
	if label==1:
		chr_count_pos[chr_num]+=1
	else:
		chr_count_neg[chr_num]+=1

print("chromosome specific positive samples")
print(chr_count_pos)
print("chromosome specific negative samples")
print(chr_count_neg)

# obtain cage sites and fasta
cage_sites_and_fasta = pd.read_csv("hg19_cage_peak_phase1and2combined_coord.csv")
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
tata_motif_dists = []
inr_motif_dists = []
dpe_motif_dists = []
upe_motif_dists = []
lowest_tata_dists = []
lowest_inr_dists = []
lowest_upe_dists = []
lowest_dpe_dists = []
tata_count = 0
inr_count = 0
dpe_count = 0
upe_count = 0
chr1 = "chr1"
chr2 = "chr2"
chr3 = "chr3" 
for ind in range(len(cage_sites_and_fasta.index)):
    chr_num = cage_sites_and_fasta["chromosomeID"].loc[ind]
    cage_start_site = cage_sites_and_fasta["start_coordinate"].loc[ind]
    cage_end_site = cage_sites_and_fasta["end_coordinate"].loc[ind]
    cage_peak = str(chr_num)+":"+str(cage_start_site)+"-"+str(cage_end_site)
    cage_begin = 500
    cage_end = 500+int(cage_end_site)-int(cage_start_site)
    near_cage_start = chr_num+":"+str(int(cage_start_site)//1000 *1000)
    if(chr_num==chr1 or chr_num==chr2 or chr_num==chr3):
            chrom_seq = genome[chr_num][int(cage_start_site)-500:int(cage_start_site)+500].seq.upper()
            seq_len = len(chrom_seq)
            if ind%100==0:
                print(chrom_seq,seq_len)
            if seq_len!=1000 or chrom_seq.find('N')!=-1:
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
                
                # Search for motifs in the chromosome
                motif_matches = search_motifs(chr_num, "TATA", chrom_seq)
                tata_dist = [absolute(500-motif_match["start"]) for motif_match in motif_matches]
                motif_matches = search_motifs(chr_num, "Inr", chrom_seq)
                inr_dist = [absolute(500-motif_match["start"]) for motif_match in motif_matches]
                motif_matches = search_motifs(chr_num, "UPE", chrom_seq)
                upe_dist = [absolute(500-motif_match["start"]) for motif_match in motif_matches]
                motif_matches = search_motifs(chr_num, "DPE", chrom_seq)
                dpe_dist = [absolute(500-motif_match["start"]) for motif_match in motif_matches]
                if len(tata_dist)>0:
                    lowest_tata_dist = np.min(np.array(tata_dist))
                    lowest_tata_dists.append(lowest_tata_dist)
                    tata_count+=1
                else:
                    lowest_tata_dists.append(500)                    
                if len(inr_dist)>0:
                    lowest_inr_dist = np.min(np.array(inr_dist))
                    lowest_inr_dists.append(lowest_inr_dist)
                else:
                    lowest_inr_dists.append(500)
                    inr_count+=1
                if len(upe_dist)>0:
                    lowest_upe_dist = np.min(np.array(upe_dist))
                    lowest_upe_dists.append(lowest_upe_dist)
                    upe_count+=1
                else:
                    lowest_upe_dists.append(500)
                if len(dpe_dist)>0:
                    lowest_dpe_dist = np.min(np.array(dpe_dist))
                    lowest_dpe_dists.append(lowest_dpe_dist)
                    dpe_count+=1
                else:
                    lowest_dpe_dists.append(500)
                tata_motif_dists.append(tata_dist)
                inr_motif_dists.append(inr_dist)
                upe_motif_dists.append(upe_dist)
                dpe_motif_dists.append(dpe_dist)


cage_sites_and_fasta2p = pd.DataFrame(data = list(zip(cage_peaks,chr_nums,cage_start_sites,cage_end_sites,
seq_lengths,seq_cage_begins,seq_cage_ends,near_cage_starts,tata_motif_dists,inr_motif_dists,upe_motif_dists,dpe_motif_dists)),
columns = ["cage_peak","chrom","cage_start","cage_end","seq_len","seq_begin","seq_end","near_cage_start",
"TATA_dist","Inr_dist","UPE_dist","DPE_dist"])
cage_sites_and_fasta2p.to_csv("cage_stats_"+str(chr1)+"_"+str(chr2)+".csv")

rows = []
avg_cage_start_deviations = []
seen_indices = []
for ind in range(len(cage_sites_and_fasta2p)):
	if ind not in seen_indices:
		row = cage_sites_and_fasta2p.loc[ind]
		near_cage_start = row["near_cage_start"]
		same_start_site_rows = cage_sites_and_fasta2p[cage_sites_and_fasta2p["near_cage_start"]==near_cage_start]
		seen_indices += same_start_site_rows.index.to_list()
		rows.append(row)
		chr_num = row["chrom"]
		cage_start_site = row["cage_start"]
		same_start_site_rows2 = same_start_site_rows.reset_index(drop=True)
		deviations = [abs(same_start_site_rows2["cage_start"].loc[ind]-cage_start_site) for ind in range(len(same_start_site_rows))]
		average_deviation = np.mean(np.array(deviations,dtype=int))
		avg_cage_start_deviations.append(average_deviation)
print(rows)
