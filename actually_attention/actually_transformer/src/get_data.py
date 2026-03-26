import pdfplumber
import os

def extract_pdf_with_debug(pdf_path, output_txt_path):
    with pdfplumber.open(pdf_path) as pdf:
        with open(output_txt_path, 'w', encoding='utf-8') as f:
            for page_num, page in enumerate(pdf.pages, 1):
                # 使用 layout=True 保留更多换行和空格
                text = page.extract_text(layout=True)
                if not text:
                    print(f"警告：第 {page_num} 页无文本")
                    continue
                
                # 统计当前页的行数
                lines = text.split('\n')
                print(f"第 {page_num} 页: {len(lines)} 行, 字符数: {len(text)}")
                if len(lines) <= 1:
                    print(f"  注意：此页只有 {len(lines)} 行，可能布局特殊，尝试用 extract_words 重建...")
                    # 如果 layout=True 仍无换行，尝试用 words 重建行
                    words = page.extract_words(keep_blank_chars=False)
                    if words:
                        # 按 y 坐标分组（同一行）
                        rows = {}
                        for w in words:
                            y0 = round(w['top'])  # 近似行坐标
                            rows.setdefault(y0, []).append(w['text'])
                        # 每行按 x 排序后合并
                        row_texts = []
                        for y in sorted(rows.keys()):
                            row = ' '.join(rows[y])
                            row_texts.append(row)
                        text = '\n'.join(row_texts)
                        lines = text.split('\n')
                        print(f"  重建后: {len(lines)} 行")
                
                f.write(text)
                # 每页之间添加两个换行分隔
                if page_num < len(pdf.pages):
                    f.write('\n\n')
    
    print(f"\n提取完成，保存至 {output_txt_path}")

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    pdf_path = os.path.normpath(os.path.join(script_dir, "../data/2025.findings-acl.198.pdf"))
    output_path = os.path.normpath(os.path.join(script_dir, "../data/extracted_text.txt"))
    
    if not os.path.exists(pdf_path):
        print(f"错误：PDF文件不存在 {pdf_path}")
    else:
        extract_pdf_with_debug(pdf_path, output_path)