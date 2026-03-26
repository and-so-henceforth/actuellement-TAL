import os
import re

def step_print(title, text, max_len=300):
    print(f"\n=== {title} ===")
    print(f"字符数: {len(text)}")
    if len(text) > 0:
        preview = text[:max_len].replace('\n', '↵')
        print(f"预览: {preview}...")
    else:
        print("警告: 文本为空！")

def remove_all_parentheses(text):
    """删除所有括号及其内容，保留换行符"""
    prev = None
    while prev != text:
        prev = text
        text = re.sub(r'\([^()]*\)', '', text, flags=re.DOTALL)
        text = re.sub(r'（[^（）]*）', '', text, flags=re.DOTALL)
        text = re.sub(r'\[[^\[\]]*\]', '', text, flags=re.DOTALL)
        text = re.sub(r'\{[^{}]*\}', '', text, flags=re.DOTALL)
    # 清理多余空格，但保留换行符
    text = re.sub(r'[^\S\n]+', ' ', text)
    return text

def remove_urls(text):
    """删除所有 URL，保留换行符"""
    url_pattern = r'https?://\S+|www\.\S+|ftp://\S+'
    text = re.sub(url_pattern, '', text, flags=re.IGNORECASE)
    text = re.sub(r'[^\S\n]+', ' ', text)
    return text

import re

def is_dirty_line(line):
    """判断一行是否应该被删除"""
    if not line:
        return True

    line_lower = line.lower()
    
    # 1. 明显的元数据标记（子串匹配，直接删除）
    strong_indicators = [
        'abstract', 'introduction', 'references', 'acknowledgments',
        'email:', '©', 'doi:', 'correspondence', 'received:', 'accepted:',
        'published:', 'funding:', 'conflict of interest', 'keywords:',
        'affiliation', 'submitted', 'conference', 'proceedings',
        'page', 'volume', 'issue', 'fig', 'figure', 'table',
        'predicted labels', 'example', 'input', 'output',
        'annual meeting', 'computational linguistics', 'acl', 'system demonstration',
        'program committee', 'association for computational linguistics',
        'order copies', 'welcome to the proceedings', 'track', 'submission', 'review',
        'acceptance rate', 'best system demonstration', 'award committee',
        'doi', 'isbn', 'issn', 'july', 'august', 'september', 'october', 'november', 'december',
        'associationforcomputational linguistics', 'programchairs', 'areachairs',
        'systemdemonstration', 'proceedings', 'aclanthology', 'order copies',
        'tel:', 'fax:', 'suite', 'street', 'blvd', 'avenue'
    ]
    if any(kw in line_lower for kw in strong_indicators):
        return True

    # 2. 机构关键词（子串匹配）
    institution_keywords = [
        'university', 'institute', 'college', 'laboratory', 'lab',
        'department', 'school', 'faculty', 'academy',
        'iit', 'iiit', 'nit', 'mit', 'stanford', 'berkeley',
        'bombay', 'mumbai', 'india'
    ]
    if any(kw in line_lower for kw in institution_keywords):
        return True

    # 3. 邮箱、星号、模型名等
    if '@' in line or '*' in line:
        return True
    if any(name in line_lower for name in ['llama', 'mistral', 'gemma', 'qwen', 'vicuna', 'gpt-']):
        return True

    # 4. 纯数字或短全大写行
    if line.isdigit():
        return True
    if line.isupper() and len(line) < 50:
        return True

    # 5. 过短行（<10字符）
    if len(line) < 10:
        return True

    # 6. 页码线：以多个点号结尾，后面可能有空格和数字
    if re.search(r'\.{3,}\s*\d+$', line):
        return True
    if re.match(r'^[\s\.]*\d+$', line):
        return True

    # 7. 作者列表特征（增强版）
    # 常见英文单词（用于降低误删正文风险）
    common_words = ['the', 'and', 'of', 'in', 'to', 'for', 'on', 'with', 'by', 'is', 'are', 'was', 'were']
    has_common = any(w in line_lower.split() for w in common_words)

    # 7.1 包含至少两个逗号，且不含常见英文单词，且大写字母数较多 → 作者列表
    if line.count(',') >= 2 and not has_common:
        upper_count = sum(1 for c in line if c.isupper())
        if upper_count >= 4:
            return True

    # 7.2 包含“and”且至少一个逗号，且不含句号（除作者名中的点号），且大写字母数较多
    if ' and ' in line_lower and line.count(',') >= 1 and '.' not in line.replace('.', ''):
        # 允许点号，但整体不能有句子结束的句号（即除名字点外无其他句号）
        if sum(1 for c in line if c.isupper()) >= 3:
            return True

    # 7.3 以逗号结尾的行，且包含至少一个逗号，不含句号，长度适中 → 可能是跨行作者列表
    if line.strip().endswith(',') and line.count(',') >= 1 and '.' not in line and len(line) < 200:
        return True

    # 7.4 纯字母、逗号、点号、短横线、空格组成的行，且至少一个逗号 → 标准作者列表
    # 放宽允许数字和空格（但数字只允许在末尾页码部分，这里先整体宽松匹配，因为后续会通过页码规则）
    if re.match(r'^[A-Za-z\s,.-]+$', line) and line.count(',') >= 1:
        return True

    # 8. 单个人名行（纯大写首字母+小写，或驼峰式）
    if re.match(r'^[A-Z][a-z]+(?:[A-Z][a-z]+)?$', line) and len(line) < 30:
        return True

    # 9. 不完整句子（以逗号或连字符结尾，且没有句号，长度适中）
    if (line.endswith(',') or line.endswith('-')) and '.' not in line and len(line) < 150:
        return True

    # 10. 包含地址模式（街道、邮编、电话）
    if re.search(r'\b(st|street|blvd|avenue|suite|tel|fax)\b', line_lower):
        return True
    if re.search(r'\b\d{5}\b', line) or re.search(r'[A-Z]{2}\s*\d{5}', line):
        return True
    if re.match(r'^\d+\s*[A-Za-z]', line) and re.search(r'(St|Street|Blvd|Ave|Dr|Rd)\.?\s*[A-Z]?', line, re.IGNORECASE):
        return True

    # 11. 以冒号结尾且很短
    if line.endswith(':') and len(line) < 30:
        return True

    # 12. 数字+空格+短单词
    if re.match(r'^[\d\s]+(Yes|No|True|False)\s*$', line, re.IGNORECASE):
        return True
    if re.match(r'^[\d\s]+$', line):
        return True

    return False

def classify_reason(line):
    line_lower = line.lower()
    if '*' in line:
        return "星号"
    if '@' in line:
        return "邮箱"
    institution_substrings = ['university', 'institute', 'college', 'lab', 'iit', 'bombay', 'mumbai', 'india']
    if any(kw in line_lower for kw in institution_substrings):
        return "机构关键词"
    meta_substrings = ['abstract', 'introduction', 'references', 'acknowledgments', 'email:', 'doi:', 'correspondence', 'keywords:', 'predicted labels', 'example', 'input', 'annual meeting', 'proceedings']
    if any(kw in line_lower for kw in meta_substrings):
        return "元数据关键词"
    model_names = ['llama', 'mistral', 'gemma', 'qwen', 'vicuna']
    if any(name in line_lower for name in model_names):
        return "模型名称"
    if len(line) < 10:
        return "过短"
    if line.isdigit() or (line.isupper() and len(line) < 50):
        return "数字/全大写"
    if re.match(r'^\d+\.?\s+[A-Z]', line) or re.match(r'^\[\d+\]', line):
        return "数字标题"
    if re.match(r'^[\s\.]+[\d]+$', line) or re.match(r'^\d+$', line):
        return "页码"
    if re.search(r'[,&]', line) and re.search(r'\.{3,}\d+$', line):
        return "人名+页码"
    if re.search(r'[A-Z][a-z]+ [A-Z][a-z]+', line) and '.' not in line and len(line) > 30:
        return "人名列表"
    if re.match(r'^[A-Z][a-zA-Z0-9]+:', line) and re.search(r'\d+$', line):
        return "论文标题+作者+页码"
    if re.search(r'\b\w+ and \w+\b', line) and re.search(r'\.{3,}\d+$', line):
        return "作者+页码"
    if re.match(r'^.*\.{3,}\d+$', line) and len(line) < 80:
        return "页码线"
    # 行末有至少三个点后跟数字，且行长度较短（说明是页码或截断）
    if re.search(r'\.{3,}\d+\s*$', line) and len(line) < 200:
        return True
    return "其他"

def clean_file(input_path, output_path):
    with open(input_path, 'r', encoding='utf-8') as f:
        raw_text = f.read()

    step_print("原始文本", raw_text)

    # 1. 删除所有括号内容
    text_no_paren = remove_all_parentheses(raw_text)
    step_print("删除所有括号后", text_no_paren)

    # 2. 删除 URL
    text_no_url = remove_urls(text_no_paren)
    step_print("删除 URL 后", text_no_url)

    # 3. 按行分割并过滤
    lines = text_no_url.split('\n')
    print(f"\n=== 按行过滤 ===")
    print(f"总行数: {len(lines)}")

    kept_lines = []
    removed_count = 0

    for idx, line in enumerate(lines, 1):
        line = line.strip()
        if not line:
            continue
        if is_dirty_line(line):
            reason = classify_reason(line)
            print(f"行 {idx}: 删除（{reason}）: {line[:80]}...")
            removed_count += 1
        else:
            kept_lines.append(line + '\n')

    # 4. 去重（基于前40字符）
    seen = set()
    unique_lines = []
    for line in kept_lines:
        stripped = line.strip()
        prefix = stripped[:40]
        if prefix not in seen:
            seen.add(prefix)
            unique_lines.append(line)
    kept_lines = unique_lines

    cleaned_text = ''.join(kept_lines)
    # 合并多余空行（连续空行只保留一个）
    cleaned_text = re.sub(r'\n\s*\n', '\n\n', cleaned_text)

    step_print("最终清洗文本", cleaned_text)
    print(f"\n清洗完成！")
    print(f"原始行数: {len(lines)}")
    print(f"删除行数: {removed_count}")
    print(f"保留行数: {len(kept_lines)}")
    print(f"输出文件: {output_path}")

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(cleaned_text)

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_path = os.path.normpath(os.path.join(script_dir, "../data/all_extracted.txt"))
    output_path = os.path.normpath(os.path.join(script_dir, "../data/cleaned_text.txt"))

    if not os.path.exists(input_path):
        print(f"错误：输入文件不存在 {input_path}")
    else:
        clean_file(input_path, output_path)