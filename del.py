import os
import shutil

folder_path = "/root/autodl-tmp/train_png"  # 替换为你要删除文件的文件夹路径
# def del (floder):
#     # 方法一：仅删除文件，不删除文件夹
#     for filename in os.listdir(folder_path):
#         file_path = os.path.join(folder_path, filename)
#         try:
#             if os.path.isfile(file_path) or os.path.islink(file_path):
#                 os.unlink(file_path)  # 删除文件
#             elif os.path.isdir(file_path):
#                 shutil.rmtree(file_path)  # 删除文件夹及其所有内容（如果文件夹里也有内容）
#         except Exception as e:
#             print(f'Failed to delete {file_path}. Reason: {e}')

#     print("文件夹中的所有文件已被删除")


# 定义源文件路径和目标文件夹路径
source_file = "/root/autodl-tmp/train/lax_P007.mat"  # 替换为你的源文件路径
destination_folder = "/root/autodl-tmp/area"  # 替换为目标文件夹路径

# 获取文件名并构造目标文件的完整路径
file_name = os.path.basename(source_file)
destination_file = os.path.join(destination_folder, file_name)

# 移动文件
shutil.copy(source_file, destination_file)

print(f"{source_file} 已成功移动到 {destination_folder}")
