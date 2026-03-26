# 临时补救脚本
import json
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.normpath(os.path.join(script_dir, "../data/cleaned_text.txt"))

with open(DATA_PATH, 'r', encoding='utf-8') as f:
    text = f.read()

chars = sorted(list(set(text)))
char2idx = {ch: i for i, ch in enumerate(chars)}
idx2char = {i: ch for i, ch in enumerate(chars)}

vocab = {
    "char2idx": char2idx,
    "idx2char": {str(k): v for k, v in idx2char.items()},  # JSON key 必须是字符串
    "vocab_size": len(chars)
}

vocab_path = os.path.join(script_dir, "../model/vocab.json")
with open(vocab_path, 'w', encoding='utf-8') as f:
    json.dump(vocab, f, ensure_ascii=False, indent=2)

print(f"词汇表已保存到 {vocab_path}，共 {len(chars)} 个字符")
