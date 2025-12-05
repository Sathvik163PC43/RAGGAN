# ============================================================
# RAG + TEXT-GAN PIPELINE (FULL CODE, NO HARDCODING, NO CIFAR-10)
# - Using ONLY queries file (Queries_clean.csv)
# - Includes GAN accuracy, generator samples, discriminator scores
# - NOW WITH ENGLISH-ONLY DECODER FOR GENERATOR OUTPUT
# ============================================================

print("--- Installing system deps & Python packages (if needed) ---")
try:
    !curl -fsSL https://ollama.com/install.sh | sh
except Exception:
    print("Could not run Ollama install script.")

try:
    !pip install -q sentence-transformers transformers gensim pandas wordfreq
except:
    print("pip install error")

print("✅ Install block finished.")

# -------------------------
# Load CSV (ONLY query column)
# -------------------------
import subprocess, time, json, os
import requests
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer, CrossEncoder, util
import torch
import torch.nn as nn
import torch.optim as optim
from collections import Counter
from gensim import downloader as api
from gensim.utils import simple_preprocess
import re
from wordfreq import zipf_frequency

CSV_PATH = "/content/drive/MyDrive/Mock/Queries_clean_latest.csv"

if not os.path.exists(CSV_PATH):
    raise RuntimeError("Queries_clean.csv NOT FOUND at specified path")

df = pd.read_csv(CSV_PATH)

if "query" not in df.columns:
    raise RuntimeError("CSV file MUST contain a column named 'query'")

queries = df["query"].dropna().astype(str).tolist()
docs = queries.copy()
print(f"Loaded {len(docs)} documents from Queries_clean.csv")

# ============================================================
# 🔥 OLLAMA BLOCK
# ============================================================
print("\n--- Starting Ollama Server ---")
try:
    process = subprocess.Popen(["ollama", "serve"],
                               stdout=subprocess.DEVNULL,
                               stderr=subprocess.DEVNULL)
    print("🟢 Starting Ollama server... waiting 10 seconds.")
    time.sleep(10)
    print("✅ Ollama server started!")
except FileNotFoundError:
    class DummyProcess:
        def terminate(self): pass
    process = DummyProcess()
    print("⚠️ Ollama not found. Local API may fail.")

def ollama_generate(model, prompt):
    url = "http://localhost:11434/api/generate"
    try:
        r = requests.post(url, json={"model":model,"prompt":prompt}, timeout=30)
        return r.json().get("response","")
    except:
        return None

# Word2Vec
print("\n--- Loading GloVe Word2Vec Model ---")
try:
    w2v_model = api.load("glove-wiki-gigaword-50")
    print("✅ Loaded Word2Vec.")
except:
    w2v_model = None

def get_word2vec_document_scores(query, documents, model):
    if model is None:
        return [0.0] * len(documents)
    def get_vec(t):
        tokens = simple_preprocess(t)
        vecs = [model.get_vector(w) for w in tokens if w in model.key_to_index]
        return np.mean(vecs, axis=0) if vecs else np.zeros(model.vector_size)
    qv = get_vec(query)
    dvs = np.array([get_vec(d) for d in documents])
    scores = np.dot(dvs, qv) / (np.linalg.norm(dvs, axis=1) * np.linalg.norm(qv) + 1e-8)
    return [float(s) if np.isfinite(s) else 0.0 for s in scores]

# ============================================================
# EMBEDDING MODELS
# ============================================================
print("Loading bi-encoder...")
bi_encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

print("Loading cross-encoder...")
cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

print("Encoding docs...")
doc_embeddings = bi_encoder.encode(docs, convert_to_tensor=True)

# ============================================================
# RETRIEVAL
# ============================================================
def retrieve_bi_encoder(query, k=20):
    q = bi_encoder.encode(query, convert_to_tensor=True)
    sc = util.cos_sim(q, doc_embeddings)[0]
    top = sc.topk(min(k, len(docs)))
    idx = top.indices.cpu().tolist()
    val = top.values.cpu().tolist()
    return list(zip(idx, val))

def rerank_cross(query, idxs):
    pairs = [[query, docs[i]] for i in idxs]
    scores = cross_encoder.predict(pairs)
    return list(scores)

def retrieve_top_k(query, k=10):
    candidate = retrieve_bi_encoder(query, k=2*k)
    cand_idxs = [i for i,_ in candidate]
    bi_map = {i:float(s) for i,s in candidate}

    cross_scores = rerank_cross(query, cand_idxs)
    cand_docs = [docs[i] for i in cand_idxs]
    w2v_scores = get_word2vec_document_scores(query, cand_docs, w2v_model)

    merged = []
    for j,i in enumerate(cand_idxs):
        merged.append((i, {
            'bi': bi_map[i],
            'cross': float(cross_scores[j]),
            'w2v': float(w2v_scores[j])
        }))

    merged = sorted(merged, key=lambda x: x[1]['cross'], reverse=True)
    topk = merged[:k]
    return [(docs[idx], sc) for idx, sc in topk]

def rag_answer(query, model="llama3.1"):
    retrieved = retrieve_top_k(query, k=20)
    ctx = "\n".join(["- " + d for d,_ in retrieved])
    prompt = f"Use only this context:\n{ctx}\n\nQuestion:{query}\nAnswer:"
    resp = ollama_generate(model, prompt)
    return resp, retrieved

# Demo
q = queries[0]
ans, retrieved = rag_answer(q)

print("\nDemo Query:", q)
print("\nRetrieved docs with scores:")
for d,s in retrieved:
    print(f"[bi={s['bi']:.4f} cross={s['cross']:.4f} w2v={s['w2v']:.4f}] {d}")
print("\nAnswer:")
print(ans)

# ============================================================
# TEXT GAN SECTION
# ============================================================
print("\n--- TEXT GAN SECTION ---\n")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def tokenize(s): return s.lower().split()

tokens = [tokenize(x) for x in docs]
counter = Counter(w for sent in tokens for w in sent)

special = ["<PAD>","<UNK>"]
vocab = special + list(counter.keys())
stoi = {w:i for i,w in enumerate(vocab)}
itos = {i:w for w,i in stoi.items()}

max_len = max(len(s) for s in tokens)

def encode(words):
    arr = [stoi.get(w, stoi["<UNK>"]) for w in words]
    return arr + [stoi["<PAD>"]] * (max_len - len(arr))

encoded_docs = [encode(x) for x in tokens]
dataset = torch.tensor(encoded_docs)
vocab_size = len(vocab)

# ============================================================
# 🔤 ENGLISH-ONLY DECODER (ADDED AS YOU REQUESTED)
# ============================================================

letters_re = re.compile(r"^[A-Za-z']+$")

def is_english_word(w, min_zipf=2.5):
    if not letters_re.match(w): return False
    try:
        return zipf_frequency(w, "en") >= min_zipf
    except:
        return False

def decode_and_filter(token_indices, itos, keep_min=1):
    words = [itos[i] for i in token_indices if itos[i] != "<PAD>"]
    filtered = [w for w in words if is_english_word(w)]
    return filtered if len(filtered) >= keep_min else words

# ============================================================
# GAN MODELS
# ============================================================

class Generator(nn.Module):
    def __init__(self, noise, seqlen, vocab):
        super().__init__()
        self.fc = nn.Linear(noise, seqlen*vocab)
        self.seqlen = seqlen
        self.vocab = vocab
    def forward(self, z):
        x = self.fc(z).view(-1, self.seqlen, self.vocab)
        p = torch.softmax(x, dim=-1)
        return torch.argmax(p, dim=-1)

class Discriminator(nn.Module):
    def __init__(self, vocab, embed_dim=64):
        super().__init__()
        self.emb = nn.Embedding(vocab, embed_dim)
        self.net = nn.Sequential(
            nn.Linear(embed_dim*max_len,256),
            nn.LeakyReLU(0.2),
            nn.Linear(256,128),
            nn.LeakyReLU(0.2),
            nn.Linear(128,1),
            nn.Sigmoid()
        )
    def forward(self,x):
        e = self.emb(x).view(x.size(0), -1)
        return self.net(e)

noise_dim = 64
G = Generator(noise_dim, max_len, vocab_size).to(device)
D = Discriminator(vocab_size).to(device)

criterion = nn.BCELoss()
optG = optim.Adam(G.parameters(), lr=0.0002)
optD = optim.Adam(D.parameters(), lr=0.0002)

print("Training GAN with Accuracy, Samples, and Scores...\n")
epochs = 50
batch = 8

for epoch in range(1, epochs+1):

    idx = torch.randint(0, len(dataset), (batch,))
    real = dataset[idx].to(device)
    real_lab = torch.ones(batch,1).to(device)
    fake_lab = torch.zeros(batch,1).to(device)

    noise = torch.randn(batch, noise_dim).to(device)
    fake = G(noise)

    lossD_real = criterion(D(real), real_lab)
    lossD_fake = criterion(D(fake.detach()), fake_lab)
    lossD = lossD_real + lossD_fake

    optD.zero_grad()
    lossD.backward()
    optD.step()

    with torch.no_grad():
        real_pred = (D(real) > 0.5).float()
        fake_pred = (D(fake) > 0.5).float()
        D_acc = ((real_pred == real_lab).float().mean() +
                 (fake_pred == fake_lab).float().mean())/2

    noise = torch.randn(batch, noise_dim).to(device)
    fake = G(noise)
    lossG = criterion(D(fake), real_lab)

    optG.zero_grad()
    lossG.backward()
    optG.step()

    if epoch % 5 == 0:
        print(f"\nEpoch {epoch}/{epochs} | D_loss={lossD:.4f} | G_loss={lossG:.4f} | D_acc={D_acc:.4f}")

        # ======================================================
        # 🟩 UPDATED — ENGLISH FILTERED GENERATOR SAMPLE
        # ======================================================
        with torch.no_grad():
            test_noise = torch.randn(1, noise_dim).to(device)
            gen_tokens = G(test_noise)[0].cpu().tolist()
            gen_words = decode_and_filter(gen_tokens, itos, keep_min=1)
            print("🟩 Generator sample:", " ".join(gen_words))

        with torch.no_grad():
            rs = D(real[:1]).item()
            fs = D(fake[:1]).item()
            print(f"🟦 Discriminator real score: {rs:.4f}")
            print(f"🟥 Discriminator fake score: {fs:.4f}")

# ============================================================
# FINAL OUTPUT — ALSO REPLACED WITH ENGLISH DECODER
# ============================================================

noise = torch.randn(5, noise_dim).to(device)
fake = G(noise).cpu().tolist()

print("\nFinal Generated Samples:")
for s in fake:
    words = decode_and_filter(s, itos, keep_min=1)
    print("•", " ".join(words))

print("\nAll modules finished successfully.")
