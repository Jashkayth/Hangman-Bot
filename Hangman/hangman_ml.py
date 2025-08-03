# hangman_ml.py
# ────────────────────────────────────────────────────────────────────
#  Machine-learned Hangman helper
#    • balanced training-set generator (3 M samples by default)
#    • 3-layer MLP (512-512-256) with dropout
#    • predict_next_letter(…) & predict_top_k(…)
# ────────────────────────────────────────────────────────────────────
import random, math, collections, time, string, torch
from bitarray import bitarray
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn, torch.optim as optim
from sklearn.model_selection import train_test_split


def load_words(path, min_len=3, max_len=20):
    with open(path, encoding="utf8") as f:
        words = [w.strip().lower() for w in f if w.strip()]
    return [w for w in words
            if min_len <= len(w) <= max_len and w.isalpha()]


def build_bitset_index(buckets):
    has = {}
    for L, wlist in buckets.items():
        has[L] = {ch: bitarray(len(wlist)) for ch in string.ascii_lowercase}
        for b in has[L].values(): b.setall(False)
        for i, w in enumerate(wlist):
            for ch in set(w): has[L][ch][i] = True
    return has
def build_lacks_index(has):
    return {L: {ch: ~ba for ch, ba in cmap.items()} for L, cmap in has.items()}


def best_split_letter(live, has_map, lacks_map, guessed):
    total = live.count()
    best, best_ent = None, -1
    for l, hbits in has_map.items():
        if l in guessed: continue
        present = (live & hbits).count()
        absent  = total - present
        if present == 0 or absent == 0: continue
        p1, p2 = present / total, absent / total
        ent = -(p1*math.log2(p1) + p2*math.log2(p2))
        if ent > best_ent: best_ent, best = ent, l
    if best: return best
    for l in string.ascii_lowercase:
        if l not in guessed: return l


def sample_windows(word, win=10, k=3):
    if len(word) <= win: return []
    starts = random.sample(range(len(word)-win+1),
                           k=min(k, len(word)-win+1))
    return [word[s:s+win] for s in starts]


def balanced_quota(words, max_len, cap):
    count = collections.Counter(len(w) for w in words)
    weight = {L: count[L]*L for L in count}        
    Z = sum(weight.values())
    raw = {L: cap * weight[L] / Z for L in weight}
    quota = {L: max(1000, int(r)) for L, r in raw.items()}
    # adjust to exact cap
    excess = sum(quota.values()) - cap
    for L in sorted(quota, key=quota.get, reverse=True):
        if excess <= 0: break
        cut = min(excess, quota[L]-1000)
        quota[L] -= cut; excess -= cut
    return quota      

def generate_samples(words, max_len, cap=3_000_000,
                     n_games=1, max_steps=7, win=10):
    quota = balanced_quota(words, max_len, cap)
    # buckets & indexes
    buckets = collections.defaultdict(list)
    for w in words: buckets[len(w)].append(w)
    has = build_bitset_index(buckets)
    lacks = build_lacks_index(has)

    all_samples = []
    for L, q in quota.items():
        src = buckets[L][:]
        if L > win:
            for w in buckets[L]:
                src.extend(sample_windows(w, win))
        random.shuffle(src)
        src = src[:q]
        for word in src:
            live = bitarray(len(buckets[len(word)])); live.setall(True)
            pat = ['_'] * len(word)
            guessed = set()
            for _ in range(max_steps):
                letter = best_split_letter(
                    live, has[len(word)], lacks[len(word)], guessed)
                all_samples.append((''.join(pat), sorted(guessed), letter))
                guessed.add(letter)
                changed = any(ch == letter for ch in word)
                for i, ch in enumerate(word):
                    if ch == letter: pat[i] = ch
                live &= has[len(word)][letter] if changed else lacks[len(word)][letter]
                if '_' not in pat or len(all_samples) >= cap: break
        if len(all_samples) >= cap: break
    random.shuffle(all_samples)
    return all_samples[:cap]


PAD='_'; CHAR2IDX={PAD:0, **{c:i+1 for i,c in enumerate(string.ascii_lowercase)}}
def enc_pattern(p, maxlen):
    return [CHAR2IDX[c] for c in p.ljust(maxlen, PAD)[:maxlen]]
def enc_guessed(g):
    v=[0]*26
    for ch in g: v[ord(ch)-97]=1
    return v
def enc_sample(sample, maxlen):
    pat, guessed, nx = sample
    return enc_pattern(pat, maxlen) + enc_guessed(guessed), ord(nx)-97
def enc_dataset(samples, maxlen):
    X, y = [], []
    for s in samples:
        xi, yi = enc_sample(s, maxlen)
        X.append(xi); y.append(yi)
    return X, y

class HangDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
    def __len__(self): return len(self.X)
    def __getitem__(self, i): return self.X[i], self.y[i]

class HangMLP(nn.Module):
    def __init__(self, inp, drop=0.25):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(inp, 512), nn.ReLU(), nn.Dropout(drop),
            nn.Linear(512, 512), nn.ReLU(), nn.Dropout(drop),
            nn.Linear(512, 256), nn.ReLU(), nn.Dropout(drop),
            nn.Linear(256, 26)
        )
    def forward(self, x): return self.net(x)


def train(model, tr_loader, va_loader, epochs=8, lr=1e-3, device="cpu"):
    model.to(device)
    opt = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    for ep in range(1, epochs+1):
        model.train(); tot=cor=0
        for Xb, yb in tr_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            opt.zero_grad(); out = model(Xb)
            loss = loss_fn(out, yb); loss.backward(); opt.step()
            tot += len(Xb); cor += (out.argmax(1) == yb).sum().item()
        tr_acc = cor / tot
        model.eval(); vtot=vcor=0
        with torch.no_grad():
            for Xb, yb in va_loader:
                Xb, yb = Xb.to(device), yb.to(device)
                out = model(Xb)
                vtot += len(Xb); vcor += (out.argmax(1) == yb).sum().item()
        print(f"ep {ep:2d}/{epochs}  train {tr_acc:.3f}  val {vcor/vtot:.3f}")


def _features(pattern, guessed, maxlen):
    return enc_pattern(pattern.lower().replace(" ", ""), maxlen) + enc_guessed(guessed)

@torch.no_grad()
def predict_next_letter(model, pattern, guessed, maxlen, device="cpu"):
    feats = torch.tensor([_features(pattern, guessed, maxlen)],
                         dtype=torch.float32).to(device)
    logits = model(feats)[0]
    for ch in guessed:
        logits[ord(ch)-97] = -1e9
    return chr(int(torch.argmax(logits)) + 97)

@torch.no_grad()
def predict_top_k(model, pattern, guessed, k, maxlen, device="cpu"):
    feats = torch.tensor([_features(pattern, guessed, maxlen)],
                         dtype=torch.float32).to(device)
    logits = model(feats)[0]
    for ch in guessed:
        logits[ord(ch)-97] = -1e9
    top = torch.topk(logits, k)
    return [chr(i+97) for i in top.indices.tolist()]

class FastFilter:
    def __init__(self, word_list):
        self.by_len = collections.defaultdict(list)
        for w in word_list:
            self.by_len[len(w)].append(w)
        self.has = build_bitset_index(self.by_len)
        self.lacks = build_lacks_index(self.has)

    def reset(self, length):
        self.length = length
        self.live = bitarray(len(self.by_len[length])); self.live.setall(True)
        self.present=set(); self.absent=set()

    def update(self, pattern, last_guess=None):
        pat = pattern.replace(" ","")
        if last_guess and last_guess not in pat:
            self.absent.add(last_guess)
        for ch in pat:
            if ch!='_': self.present.add(ch)
        L=self.length
        if last_guess:
            if last_guess in self.present:
                self.live &= self.has[L][last_guess]
            else:
                self.live &= self.lacks[L][last_guess]
        for ch in self.present:
            self.live &= self.has[L][ch]
        for ch in self.absent:
            self.live &= self.lacks[L][ch]

    def best_letter(self, guessed):
        total=self.live.count()
        best,best_ent=None,-1
        for l in string.ascii_lowercase:
            if l in guessed: continue
            present=(self.live & self.has[self.length][l]).count()
            absent=total-present
            if present==0 or absent==0: continue
            p1,p2=present/total,absent/total
            ent=-(p1*math.log2(p1)+p2*math.log2(p2))
            if ent>best_ent: best_ent,best=ent,l
        return best or next(l for l in string.ascii_lowercase if l not in guessed)

def main_fast(dictionary_path,
              max_word_len=20,
              sample_cap=3_000_000,
              batch=512,
              epochs=8,
              save="hangman_mlp_balanced.pth",
              device="cpu"):
    t0 = time.time()
    words = load_words(dictionary_path, max_len=max_word_len)
    print("words", len(words))

    samples = generate_samples(words, max_word_len, cap=sample_cap)
    print(f"samples {len(samples)}  ({time.time()-t0:.1f}s)")

    X, y = enc_dataset(samples, max_word_len)
    Xtr, Xva, ytr, yva = train_test_split(X, y, test_size=0.15, random_state=42)
    tr_loader = DataLoader(HangDataset(Xtr, ytr), batch_size=batch, shuffle=True)
    va_loader = DataLoader(HangDataset(Xva, yva), batch_size=batch*2)

    model = HangMLP(inp=len(X[0]))
    train(model, tr_loader, va_loader, epochs=epochs, device=device)
    torch.save(model.state_dict(), save)
    print("saved to", save, "  total min", (time.time()-t0)/60)

if __name__ == "__main__":
    main_fast(r"C:\Users\7ashk\Downloads\final_words.txt",
              max_word_len=20,
              sample_cap=3_000_000,
              batch=512,
              epochs=8,
              save=r"C:\Users\7ashk\Downloads\hangman_mlp_balanced.pth",
              device="cpu")   # change to "cuda" if you have GPU
