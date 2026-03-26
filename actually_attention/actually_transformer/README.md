# ActuallyTransformer

> *A Transformer too minimal to learn how to overfit.*

an extremely simplified transformer model pre-trained from scratch with vibe coding and enthusiasm

## English

### Architecture
- 1 Transformer layer  
- 1 attention head  
- Embedding dimension 64  
- Learnable positional embeddings  
- Feed‑forward: 64 → 256 → 64  
- Character‑level language model (next‑character prediction)

### Training Data
Extracted from ACL 2025 proceedings:
- Findings
- Short Papers
- Tutorials
- Industry Track
- Demo
- SRW

The cleaning script removes:
- All parentheses and their contents
- Emails, URLs
- Figure/Table captions
- Author lines (containing `*`)
- Institutional keywords, metadata
- Lines shorter than 10 characters

What remains is **a few MB of clean English text** – just enough to learn word boundaries and basic syntax.


## 中文

### 模型架构
- 1 层 Transformer  
- 1 头自注意力  
- 嵌入维度 64  
- 可学习位置编码  
- 前馈网络 64 → 256 → 64  
- 字符级语言模型（预测下一个字符）

### 训练语料
从 ACL 2025 论文集提取正文：
- Findings
- Short Papers
- Tutorials
- Industry Track
- Demo
- SRW

清洗过程删除：
- 所有括号及内容
- 邮箱、URL
- 图表标题（Figure, Table）
- 作者行（含 `*`）
- 机构关键词、元数据关键词
- 过短行（<10 字符）

最终保留 **约几 MB 纯英文正文**，刚好够模型学到单词边界和基本句法。