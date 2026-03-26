import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import re

# ========== 参数配置（集中在此，方便修改） ==========
# 模型参数
D_MODEL = 64          # 嵌入维度
D_FF = 256            # 前馈网络维度
NUM_LAYERS = 1        # Transformer 层数（极简）
NUM_HEADS = 1         # 注意力头数
MAX_LEN = 128         # 最大序列长度（上下文长度）

# 训练参数
BATCH_SIZE = 64
EPOCHS = 20
LEARNING_RATE = 3e-4
SEQ_LEN = 64          # 训练时每个样本的序列长度（建议 ≤ MAX_LEN）
TRAIN_SPLIT = 0.9     # 训练集比例（剩余用作验证）

# 数据文件路径
# 获取当前脚本所在目录（即 src/）
script_dir = os.path.dirname(os.path.abspath(__file__))
# 构建 data 文件夹下的 cleaned_text.txt 的绝对路径
DATA_PATH = os.path.normpath(os.path.join(script_dir, "../data/cleaned_text.txt"))
# 修改后的模型保存路径：向上退一级到 actually_transformer/，再进 model/
SAVE_PATH = os.path.normpath(os.path.join(script_dir, "../model/actually_transformer.pth"))

# 设备选择
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {DEVICE}")

# ========== 数据准备（字符级） ==========
class CharDataset(Dataset):
    def __init__(self, text, seq_len):
        self.seq_len = seq_len
        # 构建字符到索引的映射
        chars = sorted(list(set(text)))
        self.char2idx = {ch: i for i, ch in enumerate(chars)}
        self.idx2char = {i: ch for i, ch in enumerate(chars)}
        self.vocab_size = len(chars)
        # 将整个文本转为整数序列
        self.data = torch.tensor([self.char2idx[ch] for ch in text], dtype=torch.long)
        print(f"词汇表大小: {self.vocab_size}")
        print(f"总字符数: {len(self.data)}")

    def __len__(self):
        # 可生成的样本数 = 总字符数 - 序列长度
        return max(0, len(self.data) - self.seq_len)

    def __getitem__(self, idx):
        x = self.data[idx:idx+self.seq_len]
        y = self.data[idx+1:idx+self.seq_len+1]
        return x, y

def load_data(file_path, seq_len, train_ratio=0.9):
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    # 可选：简单清理（比如合并多余换行，但不强求）
    text = re.sub(r'\n+', '\n', text)
    dataset = CharDataset(text, seq_len)
    # 划分训练/验证
    n_train = int(len(dataset) * train_ratio)
    n_val = len(dataset) - n_train
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [n_train, n_val])
    return train_dataset, val_dataset, dataset.vocab_size, dataset.idx2char

# ========== 极简 Transformer 模型定义 ==========
class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, Q, K, V, mask=None):
        d_k = Q.size(-1)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (d_k ** 0.5)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attn = torch.softmax(scores, dim=-1)
        output = torch.matmul(attn, V)
        return output

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.attention = ScaledDotProductAttention()

    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.size()
        # 线性变换并拆分为多头
        Q = self.W_q(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        # 注意力
        attn_out = self.attention(Q, K, V, mask)
        # 合并多头
        attn_out = attn_out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        output = self.W_o(attn_out)
        return output

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
        # Pre-LN 结构
        residual = x
        x = self.norm1(x)
        x = residual + self.self_attn(x, mask)
        residual = x
        x = self.norm2(x)
        x = residual + self.feed_forward(x)
        return x

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
        # 位置编码
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(x.size(0), -1)
        x = self.token_embedding(x) + self.pos_embedding(positions)
        
        # === 修复：正确的因果掩码 ===
        # 下三角为True（当前和过去位置可见），上三角为False（未来位置mask）
        mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device)).bool()
        mask = mask.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, seq_len)
        # === 修复结束 ===
        
        for layer in self.layers:
            x = layer(x, mask)
        x = self.ln_final(x)
        logits = self.lm_head(x)
        return logits

# ========== 训练函数 ==========
def train(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for x, y in dataloader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)  # (batch, seq_len, vocab)
        loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
            total_loss += loss.item()
    return total_loss / len(dataloader)

def generate(model, start_string, idx2char, char2idx, max_new_tokens=100, temperature=1.0):
    model.eval()
    device = next(model.parameters()).device
    # 将起始字符串转为索引
    chars = [ch for ch in start_string if ch in char2idx]
    if not chars:
        return ""
    input_ids = torch.tensor([char2idx[ch] for ch in chars], device=device).unsqueeze(0)  # (1, len)
    generated = chars
    for _ in range(max_new_tokens):
        # 截取最后 MAX_LEN 个字符
        if input_ids.size(1) > MAX_LEN:
            input_ids = input_ids[:, -MAX_LEN:]
        with torch.no_grad():
            logits = model(input_ids)  # (1, seq_len, vocab)
            # 取最后一个时间步的输出
            next_token_logits = logits[0, -1, :] / temperature
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).item()
        next_char = idx2char[next_token]
        generated.append(next_char)
        input_ids = torch.cat([input_ids, torch.tensor([[next_token]], device=device)], dim=1)
    return ''.join(generated)

# ========== 主程序 ==========
if __name__ == "__main__":
    # 1. 加载数据
    print("加载数据...")
    train_dataset, val_dataset, vocab_size, idx2char = load_data(DATA_PATH, SEQ_LEN, TRAIN_SPLIT)
    # 构建 DataLoader
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=True)

    # 2. 创建模型
    model = ActuallyTransformer(vocab_size, D_MODEL, NUM_HEADS, D_FF, NUM_LAYERS, MAX_LEN).to(DEVICE)
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")

    # 3. 优化器与损失函数
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    # 4. 训练循环
    print("开始训练...")
    best_val_loss = float('inf')

    # 确保模型目录存在（第一次保存前创建）
    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)

    for epoch in range(1, EPOCHS+1):
        train_loss = train(model, train_loader, optimizer, criterion, DEVICE)
        val_loss = evaluate(model, val_loader, criterion, DEVICE)
        print(f"Epoch {epoch:2d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), SAVE_PATH)
            print(f"  模型已保存 (val_loss={val_loss:.4f})")

    # 5. 生成示例文本（从训练数据中随机取一个开头）
    print("\n生成示例文本:")
    start_text = "The"  # 你可以改成其他短前缀
    # 需要从数据集中构建 char2idx
    # 这里重新读取一下文本获取 char2idx（简单起见，用训练数据集的第一个样本的映射）
    # 更简单：直接从数据文件重建 char2idx
    with open(DATA_PATH, 'r', encoding='utf-8') as f:
        text = f.read()
    chars = sorted(list(set(text)))
    char2idx = {ch:i for i,ch in enumerate(chars)}
    generated = generate(model, start_text, idx2char, char2idx, max_new_tokens=100, temperature=0.8)
    print(f"前缀: {start_text}")
    print(f"生成: {generated}")