import os
import subprocess
import random
import sys

seq_motif = sys.argv[1]
# Define paths
jaspar_folder = "jaspar_op1k"  # Path to the folder containing 1000 .meme files
fasta_file = "motifs_with_p_vals_and_seqs_5_dist_"+seq_motif+".fa"  # Path to the RNA sequences file
fimo_output_folder = "fimo_results_motifs_20_"+seq_motif  # Output folder for FIMO results

# Create FIMO output directory if it doesn't exist
if not os.path.exists(fimo_output_folder):
    os.makedirs(fimo_output_folder)

# Get list of all .meme files in jaspar_folder
meme_files = [f for f in os.listdir(jaspar_folder) if f.endswith(".meme")]

# Run FIMO on each .meme file
for meme_file in meme_files:
    motif_path = os.path.join(jaspar_folder, meme_file)
    output_dir = os.path.join(fimo_output_folder, os.path.splitext(meme_file)[0])
    
    # Create directory for each motif's output
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Run FIMO command
    fimo_command = ["fimo", "--oc", output_dir, motif_path, fasta_file]
    try:
        subprocess.run(fimo_command, check=True)
        print(f"FIMO analysis completed for: {meme_file}")
    except subprocess.CalledProcessError as e:
        print(f"Error running FIMO for {meme_file}: {e}")

