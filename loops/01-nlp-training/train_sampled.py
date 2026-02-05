import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from collections import Counter
import os
import re
import random
import time
import argparse
from tqdm import tqdm
import sys
import multiprocessing as mp
from safetensors.torch import save_file

# ==========================================
# 1. CONFIGURATION
# ==========================================
RAW_DATA_DIR = "data/raw_parts"
SAMPLED_FILE = "data/sampled_50m.txt"
DEFAULT_MODEL_SAVE_PATH = "word2vec_sampled_50m.safetensors"

TARGET_TOTAL_WORDS = 50_000_000
VOCAB_SIZE = 100000
CONTEXT_SIZE = 5
BATCH_SIZE = 131072
EPOCHS_DEFAULT = 10
N_NEGATIVES = 15

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Global counter for progress tracking
_progress_counter = None

def _init_pool(counter):
    global _progress_counter
    _progress_counter = counter

# ==========================================
# 2. SAMPLING LOGIC
# ==========================================
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    return text

def create_balanced_sample():
    if os.path.exists(SAMPLED_FILE):
        print(f"Sampled file already exists: {SAMPLED_FILE}")
        return

    print(f"Creating balanced sample of {TARGET_TOTAL_WORDS:,} words...")
    raw_files = [f for f in os.listdir(RAW_DATA_DIR) if f.endswith('.txt')]
    if not raw_files:
        print(f"Error: No raw parts found in {RAW_DATA_DIR}")
        sys.exit(1)

    words_per_file = TARGET_TOTAL_WORDS // len(raw_files)
    
    with open(SAMPLED_FILE, 'w', encoding='utf-8') as outfile:
        for fname in raw_files:
            fpath = os.path.join(RAW_DATA_DIR, fname)
            file_size = os.path.getsize(fpath)
            
            print(f"  Sampling from {fname}...")
            words_collected = 0
            
            block_size = 1024 * 1024 # 1MB blocks
            # Jump through file to sample diversity
            # 1MB is roughly 150k words. 
            num_blocks_needed = (words_per_file // 150_000) + 1
            jump_size = file_size // max(1, num_blocks_needed)

            pbar = tqdm(total=words_per_file, unit='words', desc=f" {fname[:15]}")
            
            with open(fpath, 'r', encoding='utf-8', errors='ignore') as infile:
                for b_idx in range(num_blocks_needed):
                    if words_collected >= words_per_file: break
                    
                    # Seek and align
                    pos = b_idx * jump_size
                    infile.seek(min(pos, file_size - block_size))
                    infile.readline() # align to newline
                    
                    # Read block
                    block = infile.read(block_size)
                    if not block: break
                    
                    cleaned = clean_text(block)
                    outfile.write(cleaned + "\n")
                    
                    word_count = len(cleaned.split())
                    words_collected += word_count
                    pbar.update(word_count)
                pbar.close()

    print(f"Finished sampling. Corpus saved to {SAMPLED_FILE}")

# ==========================================
# 3. HIGH-SPEED UTILS
# ==========================================
def _tokenization_worker(args):
    file_path, start, end, vocab = args
    tokens = []
    unk_idx = vocab.get("<UNK>")
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        f.seek(start)
        if start != 0: f.readline() 
        while f.tell() < end:
            current_pos = f.tell()
            line = f.readline()
            if not line: break
            words = clean_text(line).split()
            for w in words:
                tokens.append(vocab.get(w, unk_idx))
            bytes_processed = f.tell() - current_pos
            with _progress_counter.get_lock():
                _progress_counter.value += bytes_processed
    return np.array(tokens, dtype=np.uint32)

def get_binary_tokens(data_file, vocab):
    bin_file = data_file + ".tokens.bin"
    if os.path.exists(bin_file):
        print(f"Loading tokens into RAM...")
        tokens = np.fromfile(bin_file, dtype=np.uint32)
        return tokens

    print(f"\n--- Creating binary token cache ---")
    file_size = os.path.getsize(data_file)
    num_cpus = mp.cpu_count()
    chunk_size = file_size // num_cpus
    progress_counter = mp.Value('q', 0)
    offsets = [(data_file, i*chunk_size, (i+1)*chunk_size if i<num_cpus-1 else file_size, vocab) for i in range(num_cpus)]
    
    with mp.Pool(num_cpus, initializer=_init_pool, initargs=(progress_counter,)) as pool:
        result_async = pool.map_async(_tokenization_worker, offsets)
        pbar = tqdm(total=file_size, unit='B', unit_scale=True, desc="Tokenizing")
        while not result_async.ready():
            pbar.n = progress_counter.value; pbar.refresh(); time.sleep(0.5)
        pbar.n = file_size; pbar.refresh(); pbar.close()
        results = result_async.get()
    
    all_tokens = np.concatenate(results)
    all_tokens.tofile(bin_file)
    return all_tokens

def _vocab_worker(args):
    file_path, start, end = args
    local_counts = Counter(); local_total = 0
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        f.seek(start)
        if start != 0: f.readline()
        while f.tell() < end:
            current_pos = f.tell()
            line = f.readline()
            if not line: break
            words = clean_text(line).split()
            local_counts.update(words); local_total += len(words)
            with _progress_counter.get_lock():
                _progress_counter.value += (f.tell() - current_pos)
    return local_counts, local_total

def get_vocab_and_stats(data_file, vocab_size):
    file_size = os.path.getsize(data_file)
    num_cpus = mp.cpu_count()
    chunk_size = file_size // num_cpus
    progress_counter = mp.Value('q', 0)
    print(f"\n--- Building Vocabulary ---")
    offsets = [(data_file, i*chunk_size, (i+1)*chunk_size if i<num_cpus-1 else file_size) for i in range(num_cpus)]
    with mp.Pool(num_cpus, initializer=_init_pool, initargs=(progress_counter,)) as pool:
        result_async = pool.map_async(_vocab_worker, offsets)
        pbar = tqdm(total=file_size, unit='B', unit_scale=True, desc="Scanning")
        while not result_async.ready():
            pbar.n = progress_counter.value; pbar.refresh(); time.sleep(0.5)
        pbar.n = file_size; pbar.refresh(); pbar.close()
        results = result_async.get()
    total_word_counts = Counter(); total_words = 0
    for counts, word_count in results:
        total_word_counts.update(counts); total_words += word_count
    common = total_word_counts.most_common(vocab_size - 2)
    vocab = {word: i for i, (word, _) in enumerate(common)}
    vocab["<UNK>"] = len(vocab); vocab["<PAD>"] = len(vocab)
    idx_to_word = {i: word for word, i in vocab.items()}
    vocab_counts = np.array([total_word_counts[idx_to_word[i]] for i in range(len(vocab))])
    unigram_dist = vocab_counts ** 0.75
    unigram_dist = unigram_dist / unigram_dist.sum()
    return vocab, idx_to_word, unigram_dist, total_words

class FastBatcher:
    def __init__(self, tokens, batch_size, context_size):
        self.tokens = tokens
        self.batch_size = batch_size
        self.context_size = context_size
        self.indices = np.arange(context_size, len(tokens) - context_size)
    def __len__(self):
        return len(self.indices) // self.batch_size
    def get_batches(self):
        np.random.shuffle(self.indices)
        for i in range(0, len(self.indices), self.batch_size):
            idx_batch = self.indices[i:i+self.batch_size]
            if len(idx_batch) < self.batch_size: continue
            targets = torch.from_numpy(self.tokens[idx_batch].astype(np.int64))
            contexts = []
            for j in range(-self.context_size, self.context_size + 1):
                if j == 0: continue
                contexts.append(self.tokens[idx_batch + j])
            contexts = np.stack(contexts, axis=1)
            contexts = torch.from_numpy(contexts.astype(np.int64))
            yield contexts, targets

class NoiseSampler:
    def __init__(self, weights, buffer_size=20000000):
        print(f"Pre-sampling {buffer_size/1e6:.0f}M noise words...")
        self.buffer = torch.multinomial(weights, buffer_size, replacement=True).to(device)
        self.ptr = 0; self.size = buffer_size
    def get_negatives(self, batch_size, n_negs):
        total = batch_size * n_negs
        if self.ptr + total > self.size: self.ptr = 0
        sample = self.buffer[self.ptr : self.ptr + total].view(batch_size, n_negs)
        self.ptr += total
        return sample

class CBOWModelImproved(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(CBOWModelImproved, self).__init__()
        self.in_embeddings = nn.Embedding(vocab_size, embedding_dim, sparse=True)
        self.out_embeddings = nn.Embedding(vocab_size, embedding_dim, sparse=True)
        nn.init.xavier_uniform_(self.in_embeddings.weight)
        nn.init.xavier_uniform_(self.out_embeddings.weight)
    def forward_with_loss(self, inputs, targets, negatives):
        h = torch.mean(self.in_embeddings(inputs), dim=1)
        target_embeds = self.out_embeddings(targets) 
        neg_embeds = self.out_embeddings(negatives)  
        pos_score = F.logsigmoid(torch.sum(h * target_embeds, dim=1))
        neg_score = torch.bmm(neg_embeds, h.unsqueeze(2)).squeeze(2)
        neg_score = F.logsigmoid(-neg_score) 
        return -(torch.mean(pos_score) + torch.mean(torch.sum(neg_score, dim=1)))

# ==========================================
# 4. MAIN LOGIC
# ==========================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=EPOCHS_DEFAULT)
    parser.add_argument("--save_path", type=str, default=DEFAULT_MODEL_SAVE_PATH)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    args = parser.parse_args()

    # Step 1: Create Sample
    create_balanced_sample()

    print(f"🚀 SAMPLED TURBO ENGAGED | Device: {device}")

    # Step 2: Vocab
    vocab_cache = SAMPLED_FILE + ".vocab.pth"
    if os.path.exists(vocab_cache):
        print("Loading cached vocab...")
        cache = torch.load(vocab_cache, weights_only=False)
        vocab=cache['vocab']; idx_to_word=cache['idx_to_word']; unigram_dist=cache['unigram_dist']
    else:
        vocab, idx_to_word, unigram_dist, _ = get_vocab_and_stats(SAMPLED_FILE, VOCAB_SIZE)
        torch.save({'vocab':vocab, 'idx_to_word':idx_to_word, 'unigram_dist':unigram_dist}, vocab_cache)

    # Step 3: Tokenize
    token_array = get_binary_tokens(SAMPLED_FILE, vocab)
    noise_sampler = NoiseSampler(torch.tensor(unigram_dist, dtype=torch.float))
    
    # Step 4: Train
    model = CBOWModelImproved(len(vocab), 300).to(device)
    optimizer = optim.SparseAdam(model.parameters(), lr=0.005)
    batcher = FastBatcher(token_array, args.batch_size, CONTEXT_SIZE)
    
    print(f"\n--- Starting Sampled Training (50M Words) ---")
    total_start_time = time.time()

    for epoch in range(args.epochs):
        total_loss = 0; model.train()
        i = None
        progress_bar = tqdm(batcher.get_batches(), total=len(batcher), desc=f"Epoch {epoch+1}")
        for i, (inputs, target) in enumerate(progress_bar):
            inputs, target = inputs.to(device), target.to(device)
            negatives = noise_sampler.get_negatives(inputs.size(0), N_NEGATIVES)
            optimizer.zero_grad()
            loss = model.forward_with_loss(inputs, target, negatives)
            loss.backward(); optimizer.step()
            total_loss += loss.item()
            if i % 20 == 0: progress_bar.set_postfix(loss=loss.item())

        if i is not None:
            print(f"Epoch {epoch+1} Stats: Avg Loss {total_loss/(i+1):.4f}")

    if args.save_path.endswith(".safetensors"):
        # Safetensors only supports flat tensors, so we save vocab separately if it's not already cached
        save_file(model.state_dict(), args.save_path)
        
        # Save companion vocab file for the model
        vocab_path = args.save_path.replace(".safetensors", ".vocab.pth")
        torch.save({'vocab':vocab, 'idx_to_word':idx_to_word, 'embedding_dim':300}, vocab_path)
        print(f"Model saved to {args.save_path}")
        print(f"Vocabulary saved to {vocab_path}")
    else:
        # Legacy .pth save
        torch.save({'model_state_dict':model.state_dict(), 'vocab':vocab, 'idx_to_word':idx_to_word, 'embedding_dim':300}, args.save_path)
        print(f"Training Complete! Final Model: {args.save_path}")

if __name__ == "__main__":
    main()