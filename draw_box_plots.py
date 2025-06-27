import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, precision_recall_curve, auc, f1_score

# --- CONFIG ---
FOLDS_ALL       = list(np.arange(1,22))+['X','Y']
METHODS         = ['llama-hg19-512', 'llama-hg19-1024', 'llama-mm9-512', 'rf-hg19-512', 'xgb-hg19-512', 'lgbm-hg19-512']
METRICS         = ['Accuracy', 'Precision', 'Recall', 'F1-Score']

# prepare storage
per_fold = {m: {metric: [] for metric in METRICS} for m in METHODS}


for m in METHODS:
    path = os.path.join(f"metrics_method_{m}.csv")
    if not os.path.exists(path):
        continue
    df = pd.read_csv(path)
    df.set_index('Chromosome', inplace=True, drop=True) 
    for fold in FOLDS_ALL:
        if "chr"+str(fold) not in df.index:
            continue
        scores = df.loc["chr"+str(fold)]
        # compute metrics per fold
        acc = scores["Accuracy"]
        prec = scores["Precision"]
        rec = scores["Recall"]
        f1_score = scores["F1-Score"]
        per_fold[m]['Accuracy'].append(acc)
        per_fold[m]['Precision'].append(prec)
        per_fold[m]['Recall'].append(rec)
        per_fold[m]['F1-Score'].append(f1_score)
    

# draw boxplots with overlaid points
for metric in METRICS:
    # gather data
    data = [per_fold[m][metric] for m in METHODS if per_fold[m][metric]]
    labels = [m for m in METHODS if per_fold[m][metric]]
    if not data:
        continue

    plt.figure()
    # boxplot
    plt.boxplot(data, labels=labels, widths=0.6)
    # overlay each fold as a jittered dot
    for i, vals in enumerate(data, start=1):
        jitter = np.random.normal(loc=i, scale=0.08, size=len(vals))
        plt.scatter(jitter, vals, s=40)

    plt.title(f"{metric.replace('_',' ').title()} Across Methods")
    plt.ylabel(metric.replace('_',' ').title())
    plt.xticks(rotation=30, ha='right')
    plt.tight_layout()
    plt.savefig(f"box_plot_{metric}.png")
