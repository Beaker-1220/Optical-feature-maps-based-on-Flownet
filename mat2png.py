import scipy.io as scio
import numpy as np
import h5py
import matplotlib.pyplot as plt
import os
from PIL import Image

def save_images(recon_img, output_dir, base_filename):
    os.makedirs(output_dir, exist_ok=True)

    # 遍历每个slice和时间帧保存图像
    num_slices = recon_img.shape[2]
    num_time_frames = recon_img.shape[3]

    for slice_idx in range(num_slices):
        for time_idx in range(num_time_frames):
            img = recon_img[:, :, slice_idx, time_idx]

            # 使用base_filename作为前缀生成新的文件名
            filename = f"{base_filename}_slice_{slice_idx:02d}_time_{time_idx:02d}.png"
            filepath = os.path.join(output_dir, filename)
            plt.imsave(filepath, img, cmap='gray')
            print(f"Saved {filename} to {output_dir}")


def process_mat_files(input_dir, output_dir):
    # 遍历输入目录下的所有.mat文件
    for filename in os.listdir(input_dir):
        if filename.endswith('.mat'):
            file_path = os.path.join(input_dir, filename)

            # 读取.mat文件
            mat_data = scio.loadmat(file_path)
            print(f"Keys in {filename}: {mat_data.keys()}")
            # 假设图像数据存储在'my_image_data'变量中，你需要根据具体的.mat文件结构修改此名称
            dataset = mat_data['recon_img']  # 替换为实际的数据集名称


            # 保存图像，base_filename 是 .mat 文件的名称（去掉后缀）
            base_filename = os.path.splitext(filename)[0]
            save_images(dataset, output_dir, base_filename)

            # # 删除原始.mat文件
            # try:
            #     os.remove(file_path)
            #     print(f"Deleted {filename}")
            # except Exception as e:
            #     print(f"Error deleting file {filename}: {e}")

def main():
    input_dir = '/root/autodl-tmp/area'
    output_dir = "/root/autodl-tmp/train_png"
    process_mat_files(input_dir, output_dir)
if __name__ == "__main__":
    main()