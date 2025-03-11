import pikepdf


def compress_pdf(input_path, output_path):
    with pikepdf.open(input_path) as pdf:
        # pdf.save(output_path, optimize_version=True)
        pdf.save(output_path)  # 移除不支持的参数


# 示例调用
compress_pdf("/Users/zhangsong/Documents/社工/社会工作实务初级 346P【微博：一拳大榴莲】.pdf", "./output_compressed.pdf")
