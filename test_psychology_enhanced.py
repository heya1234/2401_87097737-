import torch
import torch.nn.functional as F
import jieba
import pickle

# 导入模型类（需要确保与训练文件在同一目录）
from train_psychology_enhanced import FullDialogueModelEnhanced, DEVICE, SRC_LEN, D_MODEL, D_Z, N_ENC_LAYERS, N_DEC_LAYERS, N_HEADS

# 加载词表
with open('vocab_psych_enhanced.pkl', 'rb') as f:
    obj = pickle.load(f)
vocab = obj['vocab']
idx_to_word = obj['idx_to_word']
unk_id = vocab['<unk>']

model = FullDialogueModelEnhanced(len(vocab), D_MODEL, D_Z, N_ENC_LAYERS, N_DEC_LAYERS, N_HEADS).to(DEVICE)
model.load_state_dict(torch.load('psychology_enhanced_final.pth', map_location=DEVICE))
model.eval()

def encode_user(text):
    words = list(jieba.cut(text))
    ids = [vocab.get(w, unk_id) for w in words][:SRC_LEN]
    if len(ids) < SRC_LEN:
        ids += [vocab['<pad>']] * (SRC_LEN - len(ids))
    return torch.tensor(ids).unsqueeze(0).to(DEVICE)

def nucleus_sampling(probs, p=0.9):
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    cumulative = torch.cumsum(sorted_probs, dim=-1)
    remove = cumulative > p
    remove[:, 1:] = remove[:, :-1].clone()
    remove[:, 0] = False
    sorted_probs[remove] = 0.0
    sorted_probs /= sorted_probs.sum(dim=-1, keepdim=True)
    next_token = torch.multinomial(sorted_probs, 1)
    return sorted_indices.gather(-1, next_token)

def generate_response(model, src, max_len=300, temperature=0.6, top_p=0.9):
    with torch.no_grad():
        enc_out, z = model.encode(src)
        h_dec = None
        cur_token = torch.tensor([[unk_id]], device=DEVICE)  # 起始符
        output_ids = []
        for _ in range(max_len):
            logits, h_dec = model.decode_step(cur_token, enc_out, h_dec)
            logits = logits / temperature
            # 禁止 <unk>
            logits[:, unk_id] = -float('inf')
            probs = F.softmax(logits, dim=-1)
            # top-p 采样
            cur_token = nucleus_sampling(probs, p=top_p)
            if cur_token.item() == vocab['<eos>']:
                break
            output_ids.append(cur_token.item())
            # 反馈（修正形状）
            h_last = h_dec[-1].unsqueeze(0)  # (1, B, d)
            f_state = model.get_feedback_state(h_last)  # (n_enc_layers, B, d)
            enc_out, z = model.encode(src, z_init=f_state)
        reply = ''.join([idx_to_word.get(i, '<unk>') for i in output_ids])
        return reply

print("\n===== 增强版心理健康对话生成测试 =====")
prompts = [
    "我最近一直感到非常焦虑，但不知道原因是什么",
    "我经常失眠，有什么办法可以改善吗？",
    "和女朋友吵架了，心情很糟糕",
    "谢谢你的建议，我会尝试的"
]
for p in prompts:
    src = encode_user(p)
    gen = generate_response(model, src)
    print(f"User: {p}\nBot: {gen}\n")
