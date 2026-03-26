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

# 注意
## 训练数据的"幽灵"

由于 PDF 解析时的编码页错乱，训练数据中混入了**日文假名乱码**（如 `Microsoft Passpけoたrt`）。这些乱码（推测）来自：
- 日英跨语言论文中的真实日语片段（多语言 NLP 研究）
- PDF 字体映射错误导致的解码事故（`rt` → `た` 字形混淆）

因此，模型本质上是一个**"ACL 多语言幽灵"**：
- 能生成标准学术英语：`The proposed method achieves...`
- 也能生成日式英语胡话：`Microsoft Passport は多くの批判を受けたが state-of-the-art...`
- 偶尔会在英文单词中插入假名，产生**赛博朋克学术体**

**这不是 bug，是特征**（It's not a bug, it's a feature）。我们保留这种"数字考古层"，作为从 PDF 废墟中重建文本的见证。
