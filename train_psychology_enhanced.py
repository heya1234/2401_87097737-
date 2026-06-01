"""
增强版心理健康对话模型训练（修正后）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import jieba
import json
import os
from collections import Counter
import pickle

# ========== 配置 ==========
DATA_FILE = "psychology_10k.json"
SRC_LEN = 128
TGT_LEN = 256
CHUNK_SIZE = 16

FINETUNE_EPOCHS = 30
FINETUNE_BATCH = 32
FINETUNE_LR = 5e-4
D_MODEL = 512
D_Z = 512
N_ENC_LAYERS = 2
N_DEC_LAYERS = 2
N_HEADS = 8
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

VOCAB_FILE = "vocab_psych_enhanced.pkl"

# ========== 词汇表构建（同前） ==========
def build_vocab(jsonl_path, max_size=20000):
    counter = Counter()
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        first_char = f.read(1)
        f.seek(0)
        if first_char == '[':
            data = json.load(f)
        else:
            data = [json.loads(line) for line in f if line.strip()]

    for item in data:
        user = item.get('input', '')
        think = item.get('think', '')
        sys = item.get('content', '')
        counter.update(jieba.cut(user))
        counter.update(jieba.cut(think))
        counter.update(jieba.cut(sys))

    most_common = counter.most_common(max_size - 3)
    vocab = {word: i for i, (word, _) in enumerate(most_common)}
    vocab['<unk>'] = len(vocab)
    vocab['<eos>'] = len(vocab)
    vocab['<pad>'] = len(vocab)
    vocab['<sep>'] = len(vocab)
    idx_to_word = {i: word for word, i in vocab.items()}
    unk_id = vocab['<unk>']
    print(f"词汇表大小: {len(vocab)}")
    with open(VOCAB_FILE, 'wb') as f:
        pickle.dump({'vocab': vocab, 'idx_to_word': idx_to_word, 'unk_id': unk_id}, f)
    return vocab, idx_to_word, unk_id

def load_or_build_vocab():
    if os.path.exists(VOCAB_FILE):
        with open(VOCAB_FILE, 'rb') as f:
            obj = pickle.load(f)
        return obj['vocab'], obj['idx_to_word'], obj['unk_id']
    else:
        return build_vocab(DATA_FILE)

# ========== 数据集 ==========
class EnhancedPsychologyDataset(Dataset):
    def __init__(self, jsonl_path, vocab, src_len, tgt_len):
        self.pairs = []
        self.src_len = src_len
        self.tgt_len = tgt_len
        self.pad_id = vocab['<pad>']
        self.unk_id = vocab['<unk>']
        self.eos_id = vocab['<eos>']
        self.sep_id = vocab['<sep>']

        with open(jsonl_path, 'r', encoding='utf-8') as f:
            first_char = f.read(1)
            f.seek(0)
            if first_char == '[':
                data = json.load(f)
            else:
                data = [json.loads(line) for line in f if line.strip()]

        for item in data:
            user = item.get('input', '')
            think = item.get('think', '')
            sys = item.get('content', '')
            if not user or not sys:
                continue
            src_text = user + '|' + think
            src_words = list(jieba.cut(src_text))
            tgt_words = list(jieba.cut(sys)) + ['<eos>']

            src_ids = [vocab.get(w, self.unk_id) for w in src_words][:src_len]
            tgt_ids = [vocab.get(w, self.unk_id) for w in tgt_words][:tgt_len]
            self.pairs.append((src_ids, tgt_ids))

        print(f"加载了 {len(self.pairs)} 组带思维链的对话对")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        src, tgt = self.pairs[idx]
        return torch.tensor(src), torch.tensor(tgt)

def collate_fn(batch):
    src_list, tgt_list = zip(*batch)
    src_max = max(len(s) for s in src_list)
    tgt_max = max(len(t) for t in tgt_list)
    pad_id = vocab['<pad>']

    src_padded = torch.stack([F.pad(t, (0, src_max - len(t)), value=pad_id) for t in src_list])
    tgt_padded = torch.stack([F.pad(t, (0, tgt_max - len(t)), value=pad_id) for t in tgt_list])
    return src_padded, tgt_padded

# ========== 增强版模型 ==========
class FullDialogueModelEnhanced(nn.Module):
    def __init__(self, vocab_size, d_model, d_z, n_enc_layers, n_dec_layers, n_heads):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_enc_layers = n_enc_layers   # 保存层数以便复制

        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed   = nn.Embedding(512, d_model)

        self.encoder_gru = nn.GRU(d_model, d_z, num_layers=n_enc_layers, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d_model, num_heads=n_heads, batch_first=True)
        self.ln_cross   = nn.LayerNorm(d_model)
        self.decoder_gru = nn.GRU(d_model, d_model, num_layers=n_dec_layers, batch_first=True)
        self.output_proj = nn.Linear(d_model, vocab_size)
        self.feedback_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )

    def encode(self, src, z_init=None):
        B, L = src.shape
        pos = torch.arange(0, L, device=src.device).unsqueeze(0).expand(B, -1)
        emb = self.token_embed(src) + self.pos_embed(pos)
        if z_init is None:
            enc_out, z = self.encoder_gru(emb)
        else:
            enc_out, z = self.encoder_gru(emb, z_init)
        return enc_out, z

    def decode_chunk(self, tgt_input, enc_out, h_dec_prev=None):
        B, L = tgt_input.shape
        pos = torch.arange(0, L, device=tgt_input.device).unsqueeze(0).expand(B, -1)
        emb = self.token_embed(tgt_input) + self.pos_embed(pos)
        attn_out, _ = self.cross_attn(query=emb, key=enc_out, value=enc_out)
        emb = self.ln_cross(emb + attn_out)
        if h_dec_prev is None:
            out, h_last = self.decoder_gru(emb)
        else:
            out, h_last = self.decoder_gru(emb, h_dec_prev)
        logits = self.output_proj(out)
        return logits, h_last

    def decode_step(self, tgt_token, enc_out, h_dec_prev=None):
        B = tgt_token.size(0)
        pos = torch.tensor([[0]], device=tgt_token.device).expand(B, -1)
        emb = self.token_embed(tgt_token) + self.pos_embed(pos)
        attn_out, _ = self.cross_attn(query=emb, key=enc_out, value=enc_out)
        emb = self.ln_cross(emb + attn_out)
        if h_dec_prev is None:
            out, h_dec = self.decoder_gru(emb)
        else:
            out, h_dec = self.decoder_gru(emb, h_dec_prev)
        logits = self.output_proj(out[:, -1, :])
        return logits, h_dec

    def get_feedback_state(self, h_dec_last):
        """
        h_dec_last: 解码器的最后一层隐状态 (1, B, d_model)
        返回适合编码器初始状态的形状 (n_enc_layers, B, d_model)
        """
        # 取出最后一层状态，然后复制到所有编码器层
        f = self.feedback_proj(h_dec_last)  # (1, B, d_model)
        return f.repeat(self.n_enc_layers, 1, 1)  # (n_enc_layers, B, d_model)

# ========== 训练 ==========
def train():
    global vocab
    vocab, idx_to_word, unk_id = load_or_build_vocab()
    pad_id = vocab['<pad>']

    dataset = EnhancedPsychologyDataset(DATA_FILE, vocab, SRC_LEN, TGT_LEN)
    train_loader = DataLoader(dataset, batch_size=FINETUNE_BATCH, shuffle=True,
                              collate_fn=collate_fn, pin_memory=True, num_workers=2)

    model = FullDialogueModelEnhanced(len(vocab), D_MODEL, D_Z, N_ENC_LAYERS, N_DEC_LAYERS, N_HEADS).to(DEVICE)
    print(f"总参数量: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = optim.AdamW(model.parameters(), lr=FINETUNE_LR)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=FINETUNE_EPOCHS)
    scaler = torch.amp.GradScaler('cuda')

    for epoch in range(1, FINETUNE_EPOCHS+1):
        model.train()
        total_loss = 0.0
        for src, tgt in train_loader:
            src, tgt = src.to(DEVICE), tgt.to(DEVICE)
            B, T_src, T_tgt = src.shape[0], src.shape[1], tgt.shape[1]

            enc_out, z = model.encode(src)
            h_dec_prev = None
            loss_sum = 0.0
            chunk_count = 0

            with torch.amp.autocast('cuda'):
                for start in range(0, T_tgt, CHUNK_SIZE):
                    end = min(start + CHUNK_SIZE, T_tgt)
                    if end - start < 2:
                        break
                    chunk_input = tgt[:, start:end]
                    chunk_target = tgt[:, start+1:end+1] if end < T_tgt else tgt[:, start+1:]

                    logits, h_dec_prev = model.decode_chunk(chunk_input, enc_out, h_dec_prev)
                    shift_logits = logits[:, :-1, :].contiguous()
                    shift_targets = chunk_target[:, :shift_logits.size(1)].contiguous()
                    loss = F.cross_entropy(shift_logits.view(-1, len(vocab)),
                                           shift_targets.view(-1),
                                           ignore_index=pad_id)
                    loss_sum += loss
                    chunk_count += 1

                    # 弓状束反馈（修正：复制到编码器层数）
                    h_last = h_dec_prev[-1].unsqueeze(0)  # (1, B, d_model)
                    f_state = model.get_feedback_state(h_last)  # (n_enc_layers, B, d_model)
                    enc_out, z = model.encode(src, z_init=f_state)

                loss_avg = loss_sum / chunk_count

            optimizer.zero_grad()
            scaler.scale(loss_avg).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss_avg.item()

        avg = total_loss / len(train_loader)
        gpu_mem = torch.cuda.max_memory_allocated() / 1024**3 if torch.cuda.is_available() else 0
        print(f"Epoch {epoch:2d}  loss={avg:.4f}  GPU内存峰值={gpu_mem:.2f}GB")
        scheduler.step()

    torch.save(model.state_dict(), "psychology_enhanced_final.pth")
    print("增强版模型已保存至 psychology_enhanced_final.pth")
    return model, vocab, idx_to_word, unk_id

if __name__ == "__main__":
    model, vocab, idx_to_word, unk_id = train()
    print("\n训练完成！可以使用 test_psychology_enhanced.py 进行测试。")
