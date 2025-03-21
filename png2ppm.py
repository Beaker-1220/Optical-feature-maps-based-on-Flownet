import os
from PIL import Image

# 输入和输出文件夹路径
input_folder = '/root/autodl-tmp/train_png'  # 替换为你的输入文件夹路径
output_folder = '/root/autodl-tmp/train_png'  # 替换为你的输出文件夹路径

# 如果输出文件夹不存在，创建它
os.makedirs(output_folder, exist_ok=True)

# 遍历输入文件夹中的所有文件
for filename in os.listdir(input_folder):
    if filename.endswith('.ppm'):
        # 构造完整的输入路径
        input_path = os.path.join(input_folder, filename)
        
        # 打开 PNG 图像
        img = Image.open(input_path)
        
        # 构造输出路径，使用相同的文件名但后缀改为 .ppm
        output_path = os.path.join(output_folder, os.path.splitext(filename)[0] + '.png')
        
        # 保存为 PPM 格式
        img.save(output_path)
        # 删除原始的 PNG 文件
        os.remove(input_path)

print("转换完成！")
