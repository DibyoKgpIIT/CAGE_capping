
import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.optimizers import Nadam
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import MaxPooling1D
from tensorflow.keras.layers import UpSampling1D
from tensorflow.keras.layers import Add
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.callbacks import History
from tensorflow.keras.callbacks import Callback
import tensorflow.keras.losses as tf_loss
#import tensorflow_datasets as tfds
#import tensorflow_text as text
import tensorflow as tf
import collections
import os
import pathlib
import re
import string
import sys
import tempfile
import time
from clustering_lncrna_embeddings_functions import *
import deBrujinAdjGraph as deBAG
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score
import matplotlib.pyplot as plt
from pandas import Series
import matplotlib as mpl


#chr_num = sys.argv[1]
whole_orig = pd.read_csv("cage_combined_hg19_chromosomes_combined_1000_context_window.csv")
whole_orig = whole_orig.sample(frac=1.0).reset_index(drop=True)
#train_whole_orig = pd.read_csv("train_whole_1000.csv")
dev_whole_orig = pd.read_csv("dev_whole_continuous_1000.csv")
test_whole_orig = pd.read_csv("test_whole_continuous_1000.csv")

seq_encoding = {'1':'A','2':'C','3':'G','4':'T','0':'P','5':'N','6':'S','E':'7'}

train = pd.DataFrame()
dev = pd.DataFrame()
test = pd.DataFrame()
train_labels = []
train_sequences = []
test_labels = []
test_sequences = []
dev_labels = []
dev_sequences = []
for ind in range(len(whole_orig)):
    sequence = whole_orig["fasta_seq"].loc[ind]
    if len(sequence)==1000 and whole_orig["chrom"].loc[ind]!="chr1" and whole_orig["chrom"].loc[ind]!="chr2":
        train_labels.append(whole_orig["cage_tag"].loc[ind])
        train_sequences.append(sequence)


for ind in range(len(dev_whole_orig)):
    sequence = dev_whole_orig["fasta_seq"].loc[ind]
    if len(sequence)==1000 and dev_whole_orig["chrom"].loc[ind]=="chr1":
        dev_labels.append(dev_whole_orig["cage_tag"].loc[ind])
        dev_sequences.append(sequence)

for ind in range(len(test_whole_orig)):
    sequence = test_whole_orig["fasta_seq"].loc[ind]
    if len(sequence)==1000 and test_whole_orig["chrom"].loc[ind]=="chr2":
        test_labels.append(test_whole_orig["cage_tag"].loc[ind])
        test_sequences.append(sequence)
        
train["nt"] = train_sequences
train["label"] = train_labels

dev["nt"] = dev_sequences
dev["label"] = dev_labels

test["nt"] = test_sequences
test["label"] = test_labels

print("train dataframe size:",len(train))
print("dev dataframe size:",len(dev))
print("test dataframe size:",len(test))

train_features = train["nt"].values
train_labels = train["label"].values
print("train_features[0]",train_features[0])
dev_features = dev["nt"].values
dev_labels = dev["label"].values

test_features = test["nt"].values
test_labels = test["label"].values

# Preprocessing: Simplify DNA sequences with CountVectorizer
train_vectorizer = CountVectorizer(analyzer='char', ngram_range=(3, 3))  # Using trigrams for feature extraction
X_train_sequences = train_vectorizer.fit_transform(train['nt'])

val_vectorizer = CountVectorizer(analyzer='char', ngram_range=(3, 3))  # Using trigrams for feature extraction
X_val_sequences = val_vectorizer.fit_transform(dev['nt'])

test_vectorizer = CountVectorizer(analyzer='char', ngram_range=(3, 3))  # Using trigrams for feature extraction
X_test_sequences = test_vectorizer.fit_transform(test['nt'])


MAX_TOKENS=99

def encode_sequence(seq):
    encoding_dict = {'A': 1, 'C': 2, 'G': 3, 'T': 4, 'N': 1, 'S':4, 'W': 3, 'M':2, 'K':3, 'R': 1, 'Y':4}
    return [encoding_dict[nucleotide] for nucleotide in seq]

def create_deBrujin_embedding(x, window_length):
    # Extract sequences
    permutations = deBAG.get_all_kmer_permutations('1234', 2)
    AdjMat = np.zeros((len(x), 256+12+(window_length-1)*2), dtype=np.float32)
    for i,token in enumerate(x):
        entropy = CPF_model(token)
        if(len(token)>=window_length):
            PS_res2 = swspm(token,window_length)
            fft_out = PS_res2.tolist()
            fft_out2 = []
            for j in fft_out:
                real_val = complex(j).real
                imag_val = complex(j).imag
                fft_out2.append(real_val)
                fft_out2.append(imag_val)
            fft_out2 = np.array(fft_out2)
            sequence = encode_sequence(token)
            kmer_dict = deBAG.get_kmer_count_from_sequence(sequence, k=3, cyclic=False)
            kmer3_edges = deBAG.get_debruijn_edges_from_kmers(kmer_dict)
            deBrujin = deBAG.get_adjacency_matrix2(permutations, kmer3_edges).reshape(-1)
            combined_embedding = np.concatenate([deBrujin,entropy,fft_out2])
            AdjMat[i,:] = combined_embedding
    return AdjMat

def create_deBrujin_embedding2(x, window_length):
    # Extract sequences
    permutations = deBAG.init_all_kmer_permutation_counts('1234', 3)
    AdjMat = np.zeros((len(x), 64+12+(window_length-1)*2), dtype=np.float32)
    AdjMat2 = np.zeros((len(x), 64), dtype=np.float32)
    for i,token in enumerate(x):
        entropy = CPF_model(token)
        if(len(token)>=window_length):
            PS_res2 = swspm(token,window_length)
            fft_out = PS_res2.tolist()
            fft_out2 = []
            for j in fft_out:
                real_val = complex(j).real
                imag_val = complex(j).imag
                fft_out2.append(real_val)
                fft_out2.append(imag_val)
            fft_out2 = np.array(fft_out2)
        sequence = encode_sequence(token)
        kmer_dict = deBAG.get_kmer_count_from_sequence(sequence, k=3, cyclic=False)
        kmer_dict_embedding = [kmer_dict[key] if key in kmer_dict.keys() else permutations[key] for key in permutations.keys()]
        combined_embedding = np.concatenate([kmer_dict_embedding,entropy,fft_out2])
        AdjMat[i,:] = combined_embedding
        #AdjMat2[i,:] = kmer_dict_embedding
    return AdjMat

#def create_CNN_embeddings()
def create_deBrujin_embedding3(x, window_length):
    # Extract sequences
    permutations = deBAG.init_all_kmer_permutation_counts('1234', 3)
    AdjMat = np.zeros((len(x), 64+12+256+(window_length-1)*2), dtype=np.float32)
    AdjMat2 = np.zeros((len(x), 64), dtype=np.float32)
    for i,token in enumerate(x):
        entropy = CPF_model(token)
        if(len(token)>=window_length):
            PS_res2 = swspm(token,window_length)
            fft_out = PS_res2.tolist()
            fft_out2 = []
            for j in fft_out:
                real_val = complex(j).real
                imag_val = complex(j).imag
                fft_out2.append(real_val)
                fft_out2.append(imag_val)
            fft_out2 = np.array(fft_out2)
        sequence = encode_sequence(token)
        kmer_dict = deBAG.get_kmer_count_from_sequence(sequence, k=3, cyclic=False)
        kmer_dict_embedding = [kmer_dict[key] if key in kmer_dict.keys() else permutations[key] for key in permutations.keys()]
        kmer3_edges = deBAG.get_debruijn_edges_from_kmers(kmer_dict)
        permutations2 = deBAG.get_all_kmer_permutations('1234', 2)
        deBrujin = deBAG.get_adjacency_matrix2(permutations2, kmer3_edges).reshape(-1)
        combined_embedding = np.concatenate([kmer_dict_embedding,deBrujin,entropy,fft_out2])
        AdjMat[i,:] = combined_embedding
        #AdjMat2[i,:] = kmer_dict_embedding
    return AdjMat


def positional_encoding(length,depth):
    depth = depth/2
    positions = np.arange(length)[:,np.newaxis] #(seq,1)
    depths = np.arange(depth)[np.newaxis,:]/depth #(1,depth)
    angle_rates = 1/(10000**depths)
    angle_rads = positions * angle_rates
    pos_encoding = np.concatenate([np.sin(angle_rads),np.cos(angle_rads)],axis=-1)
    return tf.cast(pos_encoding,dtype=tf.float32)

class PositionalEmbedding(tf.keras.layers.Layer):
    def __init__(self,d_model):
        super().__init__()
        self.d_model = d_model
        self.pos_encoding = positional_encoding(length=MAX_TOKENS, depth=d_model)

    def call(self,x):
        length = tf.shape(x)[1]
        # This factor sets the relative scale of the embedding and positional_encoding.
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x = x + self.pos_encoding[tf.newaxis, :length, :]
        return x

def split_sequence(sequence,word_size):
    # print(sequence)
    words = [''.join(sequence[int(word_size//2)*i:int(word_size//2)*(i+2)]) for i in range((len(sequence)*2//word_size)-1)]
    return words,' '.join(words).strip(' ')
    
embed_nt = PositionalEmbedding(d_model=96)
#embed_op = PositionalEmbedding(vocab_size=tokenizers.op.get_vocab_size().numpy(), d_model=32)

WORD_SIZE=20
def prepare_data(nt,label,window_length):
    nt_list,nt = split_sequence(nt,WORD_SIZE)     # Output is ragged.
    nt2 = create_deBrujin_embedding3(nt_list,window_length)
    return nt, nt2, label


BUFFER_SIZE = 20000
BATCH_SIZE = 8
print("MAX_TOKENS:",MAX_TOKENS)
print("BUFFER_SIZE:",BUFFER_SIZE)
print("BATCH_SIZE:",BATCH_SIZE)

def make_dataset(df,window_length):
    nts = [] 
    features = []
    labels = []
    df_features = df["nt"].values
    df_labels = df["label"].values
    for ind in range(len(df)):
        seq = df_features[ind]
        label = df_labels[ind]
        seq,feature,label = prepare_data(seq,label,window_length)
        if(ind%1000==0):
            print(ind,feature.shape, label)
        if(len(feature)==99):
            nts.append(seq)
            features.append(feature)
            labels.append(label)
    return nts,features,labels


train_x1,train_x,train_y = make_dataset(train,11)
dev_x1, dev_x,dev_y = make_dataset(dev,11)
test_x1, test_x,test_y = make_dataset(test,11)


df_train = pd.DataFrame(data=list(zip(train_x1,train_y)),columns=["sequence","label"])
df_dev = pd.DataFrame(data=list(zip(dev_x1,dev_y)),columns=["sequence","label"])
df_test = pd.DataFrame(data=list(zip(test_x1,test_y)),columns=["sequence","label"])

df_train.to_csv("train.csv",index=False)
df_dev.to_csv("dev.csv",index=False)
df_test.to_csv("test.csv",index=False)


train_dataset = tf.data.Dataset.from_tensor_slices((train_x, train_y))
dev_dataset = tf.data.Dataset.from_tensor_slices((dev_x, dev_y))
test_dataset = tf.data.Dataset.from_tensor_slices((test_x, test_y))

train_dataset.save("train_dataset_hg19_1000_256_and_96")
dev_dataset.save("dev_dataset_hg19_1000_256_and_96")
test_dataset.save("test_dataset_hg19_1000_256_and_96")

"""


train_dataset = tf.data.Dataset.load("train_dataset_hg19_1000_96")
dev_dataset = tf.data.Dataset.load("dev_dataset_hg19_1000_96")
test_dataset = tf.data.Dataset.load("test_dataset_hg19_1000_96")
MAX_TOKENS = 99
BUFFER_SIZE = 20000
BATCH_SIZE = 8
def prepare_batch(nt,label):
    nt = nt[:, :MAX_TOKENS]    # Trim to MAX_TOKENS.
    nt = nt.to_tensor()  # Convert to 0-padded dense Tensor

    return nt,label

LSTM_UNITS = 128
FF_DIM = 96
DROPOUT_RATE = 0.2
def make_batches(ds):
    return (
        ds
        .shuffle(BUFFER_SIZE)
        .batch(BATCH_SIZE)
#        .map(prepare_batch, tf.data.AUTOTUNE)
        .prefetch(buffer_size=tf.data.AUTOTUNE))

train_batches = make_batches(train_dataset)
dev_batches = make_batches(dev_dataset)
test_batches = make_batches(test_dataset)
    
for nt, label in train_batches.take(1):
    print("nt.shape:",nt.shape)
    print("label:",label)
    break

for nt_test, test_label in test_batches.take(1):
    print("nt_test.shape:",nt_test.shape)
    print("test_label:",test_label)
    break

#nt_embed = embed_nt(nt)
#op_embed = embed_op(op)
#
#nt_embed._keras_mask
#

num_layers = 4
d_model = 96
dff = 96
num_heads = 8  
dropout_rate = 0.1


from tensorflow import keras
from keras import layers

class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super().__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = keras.Sequential(
            [layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim),]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs):
        attn_output,attention_scores = self.att(inputs, inputs,return_attention_scores=True)
        attn_output = self.dropout1(attn_output)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output)
        return self.layernorm2(out1 + ffn_output),attention_scores

embed_dim = 96  # Embedding size for each token
num_heads = 8  # Number of attention heads
ff_dim = 96  # Hidden layer size in feed forward network inside transformer

inputs = layers.Input(shape=(MAX_TOKENS,embed_dim))
embedding_layer = PositionalEmbedding(d_model=embed_dim)
nt_embed0 = embedding_layer(inputs)
transformer_block0 = TransformerBlock(embed_dim, num_heads, ff_dim)
x0,attn_scores0 = transformer_block0(nt_embed0)

# LSTM layers for sequence modeling
x1 = layers.Bidirectional(layers.LSTM(LSTM_UNITS, return_sequences=True))(x0)
x2 = layers.Bidirectional(layers.LSTM(LSTM_UNITS, return_sequences=True))(x1)
x = layers.Add()([x1,x2])
x = layers.Bidirectional(layers.LSTM(LSTM_UNITS, return_sequences=False))(x)
# Fully connected layers with Dropout
x = layers.Dropout(DROPOUT_RATE)(x)
x = layers.Dense(FF_DIM, activation="relu")(x)
x = layers.Dropout(0.1)(x)

# Output layer for binary classification
outputs = layers.Dense(1, activation="sigmoid")(x)

tf_model = keras.Model(inputs=inputs, outputs=outputs)

tf_model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3), loss="binary_crossentropy", metrics=["accuracy","precision","recall"])
tf_model.summary()
#checkpoint_read_path = "training_weights/cp-27-01-2025-transformer-lstm-mixed-model-mm9-chr_all_Embeddings-train1-02-0.714-0.817-0.507.weights.h5"
#tf_model.load_weights(checkpoint_read_path)

checkpoint_write_path="training_weights/cp-18-02-2025-transformer-lstm-mixed-model-hg19-1000-chr_all_Embeddings-train1-{epoch:02d}-{val_accuracy:.3f}-{val_precision:.3f}-{val_recall:.3f}.weights.h5"
cp = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_write_path,
            monitor='val_accuracy',
            verbose=1,
            save_best_only = True,
            save_weights_only=True,
            mode="max")

history = tf_model.fit(
    train_batches, batch_size=8, epochs=10, validation_data=dev_batches, callbacks=[cp]
)

test_accuracy = tf_model.evaluate(test_batches)
print(f"Test Accuracy: {test_accuracy}")

for ind in range(5):
    for nt_test, test_label in test_batches.take(ind):
        break
    print("test index",ind)
    predicted_output = tf_model.predict(nt_test,batch_size=1)
    predicted_output_probability = predicted_output[0]
    print("predicted_output_probability",predicted_output_probability)
    predicted_output_class = np.around(predicted_output[0])
    print("predicted_output_class",predicted_output_class)

"""
