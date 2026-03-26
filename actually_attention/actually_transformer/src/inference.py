import torch
import torch.nn as nn
import json
import os

# ========== 配置（保持与训练一致） ==========
D_MODEL = 64
D_FF = 256
NUM_LAYERS = 1
NUM_HEADS = 1
MAX_LEN = 128
SEQ_LEN = 64  # 生成时的最大上下文长度

script_dir = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.normpath(os.path.join(script_dir, "../model/actually_transformer.pth"))
VOCAB_PATH = os.path.normpath(os.path.join(script_dir, "../model/vocab.json"))

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {DEVICE}")

# ========== 模型定义（从 train.py 复制） ==========
class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, Q, K, V, mask=None):
        d_k = Q.size(-1)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (d_k ** 0.5)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attn = torch.softmax(scores, dim=-1)
        return torch.matmul(attn, V)

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model, self.num_heads = d_model, num_heads
        self.d_k = d_model // num_heads
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.attention = ScaledDotProductAttention()

    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.size()
        Q = self.W_q(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        attn_out = self.attention(Q, K, V, mask)
        attn_out = attn_out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        return self.W_o(attn_out)

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.activation = nn.ReLU()
    def forward(self, x):
        return self.linear2(self.activation(self.linear1(x)))

class TransformerLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = FeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
    def forward(self, x, mask=None):
        residual = x
        x = self.norm1(x)
        x = residual + self.self_attn(x, mask)
        residual = x
        x = self.norm2(x)
        return residual + self.feed_forward(x)

class ActuallyTransformer(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, d_ff, num_layers, max_len):
        super().__init__()
        self.d_model = d_model
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(max_len, d_model)
        self.layers = nn.ModuleList([
            TransformerLayer(d_model, num_heads, d_ff) for _ in range(num_layers)
        ])
        self.ln_final = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        seq_len = x.size(1)
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(x.size(0), -1)
        x = self.token_embedding(x) + self.pos_embedding(positions)
        mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device)).bool()
        mask = mask.unsqueeze(0).unsqueeze(0)
        for layer in self.layers:
            x = layer(x, mask)
        return self.lm_head(self.ln_final(x))

# ========== 加载模型和词汇表 ==========
print("加载词汇表...")
with open(VOCAB_PATH, 'r', encoding='utf-8') as f:
    vocab_data = json.load(f)
char2idx = vocab_data["char2idx"]
idx2char = {int(k): v for k, v in vocab_data["idx2char"].items()}
vocab_size = vocab_data["vocab_size"]

print(f"词汇表大小: {vocab_size}")

print("加载模型...")
model = ActuallyTransformer(vocab_size, D_MODEL, NUM_HEADS, D_FF, NUM_LAYERS, MAX_LEN).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()
print("模型加载完成！")

# ========== 生成函数 ==========
def generate(start_string, max_new_tokens=100, temperature=1.0):
    """输入前缀，生成后续文本"""
    # 过滤掉词汇表外的字符
    chars = [ch for ch in start_string if ch in char2idx]
    if not chars:
        return "[错误：输入包含未知字符]"
    
    input_ids = torch.tensor([char2idx[ch] for ch in chars], device=DEVICE).unsqueeze(0)
    
    generated = chars.copy()
    for _ in range(max_new_tokens):
        if input_ids.size(1) > MAX_LEN:
            input_ids = input_ids[:, -MAX_LEN:]
        
        with torch.no_grad():
            logits = model(input_ids)
            next_token_logits = logits[0, -1, :] / temperature
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).item()
        
        next_char = idx2char[next_token]
        generated.append(next_char)
        input_ids = torch.cat([input_ids, torch.tensor([[next_token]], device=DEVICE)], dim=1)
    
    return ''.join(generated)

# ========== 交互式玩法 ==========
if __name__ == "__main__":
    print("\n" + "="*50)
    print("ACL 论文风格生成器")
    print("输入前缀（英文），模型会续写 ACL 风格的文本")
    print("命令: /temp 0.8 (调节温度), /len 200 (调节长度), /quit (退出)")
    print("="*50 + "\n")
    
    temperature = 0.8
    gen_length = 100
    
    while True:
        try:
            user_input = input(">>> ").strip()
            
            if user_input.startswith("/quit"):
                break
            elif user_input.startswith("/temp"):
                temperature = float(user_input.split()[1])
                print(f"温度已设为 {temperature} (越高越随机)")
                continue
            elif user_input.startswith("/len"):
                gen_length = int(user_input.split()[1])
                print(f"生成长度已设为 {gen_length}")
                continue
            elif not user_input:
                continue
            
            print("生成中...")
            result = generate(user_input, max_new_tokens=gen_length, temperature=temperature)
            print(f"结果：{result}\n")
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"出错了: {e}")
    
    print("再见！")
