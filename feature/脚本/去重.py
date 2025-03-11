import hashlib
import os
from cedar.init import *


def calculate_md5_file(file_path):
    """计算文件的 MD5 哈希值"""
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        # 分块读取文件，以避免大文件占用过多内存
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def find_duplicate_files(directory):
    """在指定目录中查找重复文件"""
    md5_dict = {}
    duplicates = []

    # 遍历目录中的所有文件
    for root, _, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            file_md5 = calculate_md5_file(file_path)

            # 如果哈希值已存在，说明是重复文件
            if file_md5 in md5_dict:
                duplicates.append((md5_dict[file_md5], file_path))
                md5_dict[file_md5].append(file_path)
            else:
                # 存储文件路径列表，以支持多个重复文件
                md5_dict[file_md5] = [file_path]

    return duplicates


# 示例用法
if __name__ == "__main__":
    directory = input("请输入目录路径：")
    save_dir = directory + "_output"
    duplicates = find_duplicate_files(directory)

    if not duplicates:
        print("没有找到重复文件。")
    else:
        print("找到重复文件：")
        for group in duplicates:
            for file in group[1:]:
                print("重复文件：")
                print(file)
                move_file(file, save_dir)

            print("---")
