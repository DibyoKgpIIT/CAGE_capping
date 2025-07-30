import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
from pandas import Series
import os
import sys
import numpy as np

# Load the dataset
file_path = sys.argv[1]
train = pd.read_csv(os.path.join(file_path,"train.csv"))
val = pd.read_csv(os.path.join(file_path,"dev.csv"))
test = pd.read_csv(os.path.join(file_path,"test.csv"))

# Preprocessing: Simplify DNA sequences with CountVectorizer
vectorizer = CountVectorizer(analyzer='char', ngram_range=(3, 3))  # Using trigrams for feature extraction
X_train_sequences = vectorizer.fit_transform(train['sequence'])
X_val_sequences = vectorizer.fit_transform(val['sequence'])
X_test_sequences = vectorizer.fit_transform(test['sequence'])

X_train = pd.DataFrame(X_train_sequences.toarray(), columns=vectorizer.get_feature_names_out())
X_val = pd.DataFrame(X_val_sequences.toarray(), columns=vectorizer.get_feature_names_out())
X_test = pd.DataFrame(X_test_sequences.toarray(), columns=vectorizer.get_feature_names_out())

# Target variable

y_train = train["label"]
y_val = val["label"]
y_test = test["label"]

# Models
models = {
    "Random Forest": RandomForestClassifier(random_state=42),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
    "LightGBM": LGBMClassifier(random_state=42)
}

# Training and evaluation
results = {}
for name, model in models.items():
    # Train the model on the training set
    model.fit(X_train, y_train)
    
    # Evaluate on validation set
    y_val_pred = model.predict(X_val)
    val_accuracy = accuracy_score(y_val, y_val_pred)
    val_f1 = f1_score(y_val, y_val_pred)
    val_precision = precision_score(y_val, y_val_pred)
    val_recall = recall_score(y_val, y_val_pred)
    # Evaluate on test set
    y_test_pred = model.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    test_f1 = f1_score(y_test, y_test_pred)
    test_precision = precision_score(y_test, y_test_pred)
    test_recall = recall_score(y_test, y_test_pred)
    
    # Store metrics
    results[name] = {
        "Validation Accuracy": val_accuracy,
        "Validation F1-Score": val_f1,
        "Validation Precision": val_precision,
        "Validation Recall": val_recall,	
        "Test Accuracy": test_accuracy,
        "Test F1-Score": test_f1,
	"Test Precision": test_precision,
	"Test Recall": test_recall,
        "Validation Report": classification_report(y_val, y_val_pred),
        "Test Report": classification_report(y_test, y_test_pred)
    }
    
    # Print performance
    print(f"{name} Performance:")
    print(f"Validation Accuracy: {val_accuracy:.4f}, Validation F1-Score: {val_f1:.4f}, Validation Precision: {val_precision:.4f}, Validation Recall: {val_recall:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}, Test F1-Score: {test_f1:.4f}, Test Precision: {test_precision:.4f}, Test Recall: {test_recall:.4f}")

    print("\nValidation Classification Report:\n", classification_report(y_val, y_val_pred))
    print("\nTest Classification Report:\n", classification_report(y_test, y_test_pred))

# Save results summary to a CSV file
for model, metrics in results.items():
	print(model,metrics)
summary_df = pd.DataFrame({
    model: {
        "Validation Accuracy": metrics["Validation Accuracy"],
        "Validation F1-Score": metrics["Validation F1-Score"],
        "Validation Precision": metrics["Validation Precision"],
        "Validation Recall": metrics["Validation Recall"],
        "Test Accuracy": metrics["Test Accuracy"],
        "Test F1-Score": metrics["Test F1-Score"],
        "Test Precision": metrics["Test Precision"],
        "Test Recall": metrics["Test Recall"]
    }
    for model, metrics in results.items()
}).T

# Save results to files
summary_df.to_csv(file_path+"_performance_summary_with_validation2.csv")

with open(file_path+"_detailed_classification_reports.txt", "w") as f:
    for name, metrics in results.items():
        f.write(f"{name} Validation Report:\n")
        f.write(metrics["Validation Report"])
        f.write("\n\n")
        f.write(f"{name} Test Report:\n")
        f.write(metrics["Test Report"])
        f.write("\n\n")


# Feature importance visualization 

for model_name in ("XGBoost","Random Forest","LightGBM"):
    selected_model = models[model_name]  
    selected_model.fit(X_test, y_test)

    plt.figure(figsize=(12, 3))
    features = X_test.columns.values.tolist()
    importance = selected_model.feature_importances_.tolist()
    mean = np.mean(np.array(importance,dtype=float))
    std = np.std(np.array(importance,dtype=float))
    print(model_name+"_mean",mean)
    print(model_name+"_std",std)
    feature_series = Series(data=importance, index=features)

    feature_series.plot.bar()
    plt.title('Feature Importance')
    plt.xlabel('Features')
    plt.ylabel('Importance Score')
    plt.tight_layout()
    plt.savefig(file_path+"_"+model_name+"_feature_importance.png")

