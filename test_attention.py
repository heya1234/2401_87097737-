"""
带注意力机制的 Seq2Seq 测试
"""

import torch
import torch.nn.functional as F
import jieba
import pickle
from train_attention import AttnSeq2Seq, DEVICE, SRC_LEN, TGT_LEN, D_MODEL, VOCAB_FILE

with open(VOCAB_FILE, 'rb') as f:
    obj = pickle.load(f)
vocab = obj['vocab']
idx_to_word = obj['idx_to_word']
unk_id = vocab['<unk>']

model = AttnSeq2Seq(len(vocab), D_MODEL).to(DEVICE)
model.load_state_dict(torch.load("attention_model.pth", map_location=DEVICE))
model.eval()

def encode(text):
    words = list(jieba.cut(text))
    ids = [vocab.get(w, unk_id) for w in words][:SRC_LEN]
    ids += [vocab['<pad>']] * (SRC_LEN - len(ids))
    return torch.tensor(ids).unsqueeze(0).to(DEVICE)

def generate(user_text, max_len=TGT_LEN, temperature=0.8):
    src = encode(user_text)
    src_emb = model.embed(src)
    enc_out, h_n = model.encoder(src_emb)

    h_dec = h_n
    cur_token = torch.tensor([[unk_id]], device=DEVICE)
    output_ids = []
    for _ in range(max_len):
        cur_emb = model.embed(cur_token)   # (1, 1, d)
        # 注意力
        attn_weights = model._attention(h_dec[-1], enc_out)  # (B, S)
        context = torch.bmm(attn_weights.unsqueeze(1), enc_out)
        rnn_input = torch.cat([cur_emb, context], dim=-1)
        out, h_dec = model.decoder(rnn_input, h_dec)
        logits = model.out(torch.cat([out, context], dim=-1)).squeeze(1) / temperature
        logits[:, unk_id] = -float('inf')
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, 1)
        if next_token.item() == vocab['<eos>']:
            break
        output_ids.append(next_token.item())
        cur_token = next_token
    reply = ''.join([idx_to_word.get(i, '<unk>') for i in output_ids])
    return reply

print("===== 带注意力 Seq2Seq 测试 =====")
while True:
    user = input("\n用户: ")
    if user.lower() in ['exit', 'quit']:
        break
    print(f"助手: {generate(user)}")
