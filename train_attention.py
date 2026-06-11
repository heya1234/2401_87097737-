"""
带 Bahdanau 注意力的 Seq2Seq 模型
编码器：单层 GRU
解码器：单层 GRU + 加性注意力 + 输出投影
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
BATCH_SIZE = 32
EPOCHS = 30
LR = 1e-3
D_MODEL = 512
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
VOCAB_FILE = "vocab_attention.pkl"

# ========== 词汇表构建 ==========
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
        counter.update(jieba.cut(item.get('input', '')))
        counter.update(jieba.cut(item.get('content', '')))
    most_common = counter.most_common(max_size - 3)
    vocab = {word: i for i, (word, _) in enumerate(most_common)}
    vocab['<unk>'] = len(vocab)
    vocab['<eos>'] = len(vocab)
    vocab['<pad>'] = len(vocab)
    idx_to_word = {i: word for word, i in vocab.items()}
    unk_id = vocab['<unk>']
    with open(VOCAB_FILE, 'wb') as f:
        pickle.dump({'vocab': vocab, 'idx_to_word': idx_to_word, 'unk_id': unk_id}, f)
    return vocab, idx_to_word, unk_id

def load_or_build_vocab():
    if os.path.exists(VOCAB_FILE):
        with open(VOCAB_FILE, 'rb') as f:
            obj = pickle.load(f)
        return obj['vocab'], obj['idx_to_word'], obj['unk_id']
    return build_vocab(DATA_FILE)

# ========== 数据集 ==========
class AttentionDataset(Dataset):
    def __init__(self, jsonl_path, vocab):
        self.pairs = []
        self.pad_id = vocab['<pad>']
        self.unk_id = vocab['<unk>']
        self.eos_id = vocab['<eos>']
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            first_char = f.read(1)
            f.seek(0)
            if first_char == '[':
                data = json.load(f)
            else:
                data = [json.loads(line) for line in f if line.strip()]
        for item in data:
            src = item.get('input', '')
            tgt = item.get('content', '')
            if not src or not tgt:
                continue
            src_ids = [vocab.get(w, self.unk_id) for w in jieba.cut(src)][:SRC_LEN]
            tgt_ids = [vocab.get(w, self.unk_id) for w in jieba.cut(tgt)] + [self.eos_id]
            tgt_ids = tgt_ids[:TGT_LEN]
            self.pairs.append((src_ids, tgt_ids))
        print(f"加载了 {len(self.pairs)} 条样本")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        src, tgt = self.pairs[idx]
        return torch.tensor(src, dtype=torch.long), torch.tensor(tgt, dtype=torch.long)

def collate_fn(batch):
    src_list, tgt_list = zip(*batch)
    pad_id = vocab['<pad>']
    src_padded = nn.utils.rnn.pad_sequence(src_list, batch_first=True, padding_value=pad_id)[:, :SRC_LEN]
    tgt_padded = nn.utils.rnn.pad_sequence(tgt_list, batch_first=True, padding_value=pad_id)[:, :TGT_LEN]
    return src_padded, tgt_padded

# ========== 带注意力的 Seq2Seq ==========
class AttnSeq2Seq(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.embed = nn.Embedding(vocab_size, d_model)
        self.encoder = nn.GRU(d_model, d_model, batch_first=True)
        self.decoder = nn.GRU(d_model + d_model, d_model, batch_first=True)  # 输入包含上下文向量
        self.Wa = nn.Linear(d_model * 2, d_model)     # 注意力参数
        self.va = nn.Linear(d_model, 1, bias=False)
        self.out = nn.Linear(d_model * 2, vocab_size) # 输出时拼接上下文向量

    def forward(self, src, tgt, teacher_forcing=True):
        # 编码
        src_emb = self.embed(src)
        enc_out, h_n = self.encoder(src_emb)   # enc_out: (B, S, d), h_n: (1, B, d)

        B, T = tgt.shape[0], tgt.shape[1] if teacher_forcing else TGT_LEN
        # 准备解码
        if teacher_forcing:
            dec_input = self.embed(tgt[:, :-1])   # (B, T-1, d)
        else:
            # 非 teacher forcing 用于推理，此处省略
            pass

        h_dec = h_n   # (1, B, d) → 扩展为 decoder 初始状态
        dec_outputs = []
        for t in range(dec_input.size(1)):
            cur_input = dec_input[:, t, :].unsqueeze(1)   # (B, 1, d)
            # 计算注意力
            attn_weights = self._attention(h_dec[-1], enc_out)  # (B, S)
            context = torch.bmm(attn_weights.unsqueeze(1), enc_out)  # (B, 1, d)
            # 拼接输入和上下文
            rnn_input = torch.cat([cur_input, context], dim=-1)  # (B, 1, 2d)
            out, h_dec = self.decoder(rnn_input, h_dec)
            # 输出
            logits = self.out(torch.cat([out, context], dim=-1))  # (B, 1, vocab)
            dec_outputs.append(logits)

        logits = torch.cat(dec_outputs, dim=1)   # (B, T-1, vocab)
        return logits

    def _attention(self, decoder_hidden, encoder_outputs):
        """
        decoder_hidden: (B, d)
        encoder_outputs: (B, S, d)
        返回: (B, S) 注意力权重
        """
        B, S, _ = encoder_outputs.shape
        # 重复 decoder 状态 S 次
        decoder_hidden = decoder_hidden.unsqueeze(1).repeat(1, S, 1)  # (B, S, d)
        energy = torch.tanh(self.Wa(torch.cat([decoder_hidden, encoder_outputs], dim=-1)))
        attn_scores = self.va(energy).squeeze(-1)  # (B, S)
        return F.softmax(attn_scores, dim=-1)

# ========== 训练 ==========
def train():
    global vocab
    vocab, idx_to_word, unk_id = load_or_build_vocab()
    pad_id = vocab['<pad>']

    dataset = AttentionDataset(DATA_FILE, vocab)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                        collate_fn=collate_fn, pin_memory=True, num_workers=2)

    model = AttnSeq2Seq(len(vocab), D_MODEL).to(DEVICE)
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = optim.AdamW(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    scaler = torch.amp.GradScaler('cuda')

    for epoch in range(1, EPOCHS+1):
        model.train()
        total_loss = 0.0
        for src, tgt in loader:
            src, tgt = src.to(DEVICE), tgt.to(DEVICE)
            with torch.amp.autocast('cuda'):
                logits = model(src, tgt, teacher_forcing=True)
                tgt_labels = tgt[:, 1:]  # 目标
                loss = F.cross_entropy(logits.reshape(-1, len(vocab)),
                                       tgt_labels.reshape(-1),
                                       ignore_index=pad_id)
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch:2d}  Loss={avg_loss:.4f}")
        scheduler.step()

    torch.save(model.state_dict(), "attention_model.pth")
    print("模型已保存至 attention_model.pth")
    return model, vocab, idx_to_word, unk_id

if __name__ == "__main__":
    train()
