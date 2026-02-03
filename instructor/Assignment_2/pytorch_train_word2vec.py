import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from tqdm import tqdm

# Hyperparameters
EMBEDDING_DIM = 100
BATCH_SIZE = 512  # following announcement update
EPOCHS = 5 # following announcement update
LEARNING_RATE = 0.01
NEGATIVE_SAMPLES = 5  # Number of negative samples per positive

# Custom Dataset for Skip-gram
class SkipGramDataset(Dataset):
    def __init__(self, skipgram_df):
        self.center_words = torch.tensor(skipgram_df['center'].values, dtype = torch.long)
        self.context_words = torch.tensor(skipgram_df['context'].values, dtype = torch.long)

    def __len__(self):
        return len(self.center_words)
    
    def __getitem__(self, idx):
        return self.center_words[idx], self.context_words[idx]
    
# Simple Skip-gram Module
class Word2Vec(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(Word2Vec, self).__init__()
        self.center_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.context_embeddings = nn.Embedding(vocab_size, embedding_dim)

        self.center_embeddings.weight.data.uniform_(-0.5/embedding_dim, 0.5/embedding_dim)
        self.context_embeddings.weight.data.zero_()

    def forward(self, center, context, negative_samples):
        emb_center = self.center_embeddings(center)
        emb_context = self.context_embeddings(context)
        emb_negative = self.context_embeddings(negative_samples)

        pos_score = torch.sum(emb_center * emb_context, dim = 1)
        pos_loss = torch.log(torch.sigmoid(pos_score) + 1e-10)

        neg_score = torch.bmm(emb_negative, emb_center.unsqueeze(2)).squeeze()
        neg_loss = torch.sum(torch.log(torch.sigmoid(-neg_score) +1e-10), dim = 1)

        return -(pos_loss + neg_loss).mean()

    def get_embeddings(self):
        return self.center_embeddings.weight.data.cpu().numpy()

# Load processed data
with open('processed_data.pkl', 'rb') as f:
    data = pickle.load(f)

# Precompute negative sampling distribution below
vocab_size = len(data['word2idx'])
counts = torch.tensor([data['counter'][data['idx2word'][i]] for i in range(vocab_size)], dtype = torch.float)

pow_counts = torch.pow(counts, 0.75)
neg_sampling_prob = pow_counts / torch.sum(pow_counts)
# Device selection: CUDA > MPS > CPU * only CUDA possible
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
neg_sampling_prob = neg_sampling_prob.to(device)

# Dataset and DataLoader
dataset = SkipGramDataset(data['skipgram_df'])
dataloader = DataLoader(dataset, batch_size = BATCH_SIZE, shuffle = True)

# Model, Loss, Optimizer
model = Word2Vec(vocab_size, EMBEDDING_DIM).to(device)
optimizer = optim.Adam(model.parameters(), lr = LEARNING_RATE)

# Training loop
model.train()
for epoch in range(EPOCHS):
    total_loss = 0
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1} / {EPOCHS}")

    for center, context in progress_bar:
        center, context = center.to(device), context.to(device)
        
        batch_size = center.size(0)
        neg_samples = torch.multinomial(neg_sampling_prob, batch_size * NEGATIVE_SAMPLES, replacement = True)
        neg_samples = neg_samples.view(batch_size, NEGATIVE_SAMPLES).to(device)

        mask = neg_samples == context.unsqueeze(1)
        while mask.any():
            new_samples = torch.multinomial(neg_sampling_prob, mask.sum().item(), replacement=True).to(device)
            neg_samples[mask] = new_samples
            mask = neg_samples == context.unsqueeze(1)

        optimizer.zero_grad()
        loss = model(center, context, neg_samples)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        progress_bar.set_postfix(loss = loss.item())

    print('Average Loss = ', total_loss / len(dataloader))

# Save embeddings and mappings
embeddings = model.get_embeddings()
with open('word2vec_embeddings.pkl', 'wb') as f:
    pickle.dump({'embeddings': embeddings, 'word2idx': data['word2idx'], 'idx2word': data['idx2word']}, f)
print("Embeddings saved to word2vec_embeddings.pkl")
