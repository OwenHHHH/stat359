#!/usr/bin/env python
# coding: utf-8

#Task 1
# importing required libraries
import numpy as np
import pandas as pd
import datasets
import torch
import torch.nn as nn
import torch.optim as optim
import os
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from tqdm import tqdm
from gensim.models import KeyedVectors
import gensim.downloader as api

# setting values for MLP
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
NUM_EPOCHS = 30
HIDDEN_DIM = 128
DROPOUT = 0.5

os.makedirs("outputs", exist_ok=True)

def get_device():
    """Returns the appropriate device (MPS, CUDA, or CPU)."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")

# Loading model and data
dataset = datasets.load_dataset('financial_phrasebank', 'sentences_50agree', trust_remote_code=True)
data = pd.DataFrame(dataset['train'])
y = data['label'].values

# Ensure the model is downloaded via your download script before running this
if __name__ == "__main__":
    if not os.path.exists(f"fasttext-wiki-news-subwords-300.model"):
        model = api.load("fasttext-wiki-news-subwords-300")
        model.save("fasttext-wiki-news-subwords-300.model")
fasttext = KeyedVectors.load(f"fasttext-wiki-news-subwords-300.model")
# function similar to previous assignment 
def sentence_to_mean_embedding(sentence, model):
    # iterate through all of the words in the sentence and get list of context vectors for each word 
    words = sentence.lower().split()
    vectors = []
    for w in words:
        try:
            vectors.append(model[w])
        except KeyError:
            continue # Skip unknown words
    # If no words found, return zero vector     
    if not vectors:
        return np.zeros(model.vector_size)
    # return mean context vector for the sentence
    return np.mean(vectors, axis=0)

# Compute embeddings for all sentences
X_embeddings = np.array([sentence_to_mean_embedding(s, fasttext) for s in data['sentence']])
print(f"Embedding shape: {X_embeddings.shape}") # confirming embedding shape

# splitting embeddings data - given proportion of -.15, using set random state for reproducibility
# splitting into trainval and test
X_trainval, X_test, y_trainval, y_test = train_test_split(
    X_embeddings, y, test_size=0.15, stratify=y, random_state=42
)

# splitting trainval into train and val set 
X_train, X_val, y_train, y_val = train_test_split(
    X_trainval, y_trainval, test_size=0.15, stratify=y_trainval, random_state=42
)

# building dataset and loading data
class SentimentDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
    def __len__(self):
        return len(self.y)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
train_loader = DataLoader(SentimentDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(SentimentDataset(X_val, y_val), batch_size=BATCH_SIZE, shuffle=False) # shuffle off for consistent eval
test_loader = DataLoader(SentimentDataset(X_test, y_test), batch_size=BATCH_SIZE, shuffle=False)

# building MLP classifier
class MLPClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, dropout_rate):
        super(MLPClassifier, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, num_classes)
        )
    def forward(self, x):
        return self.network(x)
# Initialize Model
device = get_device()
input_dim = 300 # FastText dimension
model = MLPClassifier(input_dim, HIDDEN_DIM, 3, DROPOUT).to(device)

# calculating class counts
class_counts = np.bincount(y_train)
total_samples = len(y_train)
class_weights = 1. / class_counts
class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)
# setting loss function to crossentropy loss as instructed
criterion = nn.CrossEntropyLoss(weight=class_weights)
# setting optimizer as adam with scheduling
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)

# Training the MLP
train_loss_hist, val_loss_hist = [], [] # initializing storage
train_acc_hist, val_acc_hist = [], []
train_f1_hist, val_f1_hist = [], []
best_val_f1 = 0.0
print("Starting Training")
for epoch in range(NUM_EPOCHS): # going through all set epochs
    model.train()
    running_loss = 0.0
    all_preds, all_targets = [], []
    
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_targets.extend(labels.cpu().numpy())
        
    epoch_loss = running_loss / len(X_train)
    epoch_acc = np.mean(np.array(all_preds) == np.array(all_targets))
    epoch_f1 = f1_score(all_targets, all_preds, average='macro')
    
    train_loss_hist.append(epoch_loss)
    train_acc_hist.append(epoch_acc)
    train_f1_hist.append(epoch_f1)
    
    # Validation
    model.eval()
    val_running_loss = 0.0
    val_preds, val_targets = [], []
    
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            val_running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            val_preds.extend(preds.cpu().numpy())
            val_targets.extend(labels.cpu().numpy())
            
    val_loss = val_running_loss / len(X_val)
    val_acc = np.mean(np.array(val_preds) == np.array(val_targets))
    val_f1 = f1_score(val_targets, val_preds, average='macro')
    
    val_loss_hist.append(val_loss)
    val_acc_hist.append(val_acc)
    val_f1_hist.append(val_f1)
    
    scheduler.step(val_f1)
    
    print(f"Epoch {epoch+1}/{NUM_EPOCHS} | Train Loss: {epoch_loss:.4f} F1: {epoch_f1:.4f} | Val Loss: {val_loss:.4f} F1: {val_f1:.4f}")
    
    # Save Best Model
    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        torch.save(model.state_dict(), "outputs/best_mlp_model.pth")

# plotting required plots
epochs_range = range(1, NUM_EPOCHS + 1)
# Plot Loss, Accuracy, F1
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
# Loss
axes[0].plot(epochs_range, train_loss_hist, label='Train Loss')
axes[0].plot(epochs_range, val_loss_hist, label='Val Loss')
axes[0].set_title('Loss Curve')
axes[0].set_xlabel('Epochs')
axes[0].set_ylabel('Loss')
axes[0].legend()
# Accuracy
axes[1].plot(epochs_range, train_acc_hist, label='Train Acc')
axes[1].plot(epochs_range, val_acc_hist, label='Val Acc')
axes[1].set_title('Accuracy Curve')
axes[1].set_xlabel('Epochs')
axes[1].set_ylabel('Accuracy')
axes[1].legend()
# F1
axes[2].plot(epochs_range, train_f1_hist, label='Train F1')
axes[2].plot(epochs_range, val_f1_hist, label='Val F1')
axes[2].set_title('Macro F1 Curve')
axes[2].set_xlabel('Epochs')
axes[2].set_ylabel('F1 Score')
axes[2].legend()
# plotting
plt.tight_layout()
plt.savefig("outputs/mlp_f1_learning_curves.png")
plt.show()

# additional accuracy plot to match given python scripts
plt.figure(figsize=(8, 6))
plt.plot(train_acc_hist, label='Train Accuracy')
plt.plot(val_acc_hist, label='Val Accuracy')
plt.title('MLP Accuracy Curve')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.savefig('outputs/mlp_accuracy_learning_curve.png')

# evaluating on test set
model.load_state_dict(torch.load("outputs/best_mlp_model.pth"))
model.eval()
# predicting values
test_preds, test_targets = [], []
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        test_preds.extend(preds.cpu().numpy())
        test_targets.extend(labels.cpu().numpy())
# calculating f1 score and reporting it
test_f1 = f1_score(test_targets, test_preds, average='macro')
print(f"Test Macro F1 Score: {test_f1:.4f}")
print("Classification Report:")
print(classification_report(test_targets, test_preds, target_names=['Negative', 'Neutral', 'Positive']))

# Plotting confusion matrix
cm = confusion_matrix(test_targets, test_preds)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Neg', 'Neu', 'Pos'], yticklabels=['Neg', 'Neu', 'Pos'])
plt.title('MLP Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.savefig("outputs/mlp_confusion_matrix.png")
plt.show()