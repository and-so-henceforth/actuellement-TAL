import os
import glob
from get_data import extract_pdf_with_debug

def batch_extract(pdf_folder, output_file):
    pdf_files = glob.glob(os.path.join(pdf_folder, "*.pdf"))
    with open(output_file, 'w', encoding='utf-8') as out_f:
        for pdf_path in pdf_files:
            print(f"正在处理: {pdf_path}")
            # 临时文件，每篇单独提取后追加
            temp_file = pdf_path + ".txt"
            extract_pdf_with_debug(pdf_path, temp_file)
            with open(temp_file, 'r', encoding='utf-8') as temp_f:
                out_f.write(temp_f.read())
                out_f.write("\n\n")   # 篇间分隔
            os.remove(temp_file)
    print(f"全部提取完成，合并至 {output_file}")

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    pdf_folder = os.path.normpath(os.path.join(script_dir, "../data"))
    output_file = os.path.normpath(os.path.join(script_dir, "../data/all_extracted.txt"))
    batch_extract(pdf_folder, output_file)