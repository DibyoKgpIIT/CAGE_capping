import itertools
import numpy as np

def get_kmer_count_from_sequence(sequence, k=3, cyclic=True):
    """
    Returns dictionary with keys representing all possible kmers in a sequence
    and values counting their occurrence in the sequence.
    """
    # dict to store kmers
    kmers = {}
    if(type(sequence)==type([])):
        sequence2 = ""
        for ind in range(len(sequence)):
            sequence2 += str(sequence[ind])
    else:
        sequence2 = sequence
    # count how many times each occurred in this sequence (treated as cyclic)
    for i in range(0, len(sequence)):
        kmer = sequence2[i:i + k]
        
        # for cyclic sequence get kmers that wrap from end to beginning
        length = len(kmer)
        if cyclic:
            if len(kmer) != k:
                kmer += sequence2[:(k - length)]
        
        # if not cyclic then skip kmers at end of sequence
        else:
            if len(kmer) != k:
                continue
        
        # count occurrence of this kmer in sequence
        if kmer in kmers:
            kmers[kmer] += 1
        else:
            kmers[kmer] = 1
    
    return kmers


def get_debruijn_edges_from_kmers(kmers):
    """
    Every possible (k-1)mer (n-1 suffix and prefix of kmers) is assigned
    to a node, and we connect one node to another if the (k-1)mer overlaps 
    another. Nodes are (k-1)mers, edges are kmers.
    """
    # store edges as tuples in a set
    edges = set()
    
    # compare each (k-1)mer
    for k1 in kmers:
        for k2 in kmers:
            if k1 != k2:            
                # if they overlap then add to edges
                if k1[1:] == k2[:-1]:
                    edges.add((k1[:-1], k2[:-1]))
                if k1[:-1] == k2[1:]:
                    edges.add((k2[:-1], k1[:-1]))

    return edges

def get_all_kmer_permutations(alphabet,word_size):
    return [''.join(p) for p in itertools.product(alphabet, repeat=word_size)]       

def init_all_kmer_permutation_counts(alphabet,word_size):
    kmers = [''.join(p) for p in itertools.product(alphabet, repeat=word_size)]       
    kmers_count = {}
    for kmer in kmers:
        kmers_count[kmer] = 0
    return kmers_count

def get_adjacency_matrix(kmers,edges):
    kmers_index = {}
    for ind,kmer in enumerate(kmers):
        kmers_index[kmer] = ind
        print(kmer,kmers_index[kmer])
    adj_matrix = np.zeros((len(kmers),len(kmers)),dtype=int)
    for edge in edges:
        adj_matrix[kmers_index[edge[0]],kmers_index[edge[1]]] += 1
    return adj_matrix

def get_adjacency_matrix2(kmer_permutations,edges):
    kmers_index = {}
    for ind,permutation in enumerate(kmer_permutations):
        kmers_index[permutation] = ind
    adj_matrix = np.zeros((len(kmer_permutations),len(kmer_permutations)),dtype=int)
    for edge in edges:
        adj_matrix[kmers_index[edge[0]],kmers_index[edge[1]]] += 1
    return adj_matrix
        

