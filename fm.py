import cv2
import os
import re
from os.path import join
import numpy as np
from tqdm import tqdm

# # 读取图像
# img_t_minus_1 = cv2.imread("/root/autodl-tmp/train_png/lax_P001_slice_00_time_00.png", cv2.IMREAD_GRAYSCALE)
# img_t = cv2.imread("/root/autodl-tmp/train_png/lax_P001_slice_00_time_01.png", cv2.IMREAD_GRAYSCALE)

# 读取光流文件 O1
def read_flo_file(file_path):
    with open(file_path, 'rb') as f:
        # 读取魔数，确保文件是有效的 .flo 文件
        header = np.fromfile(f, np.float32, count=1)
        if header[0] != 202021.25:
            raise ValueError("Not a valid .flo file.")
        
        # 读取宽度和高度，这里应该用 int32 类型
        width = np.fromfile(f, np.int32, count=1)[0]
        height = np.fromfile(f, np.int32, count=1)[0]
        
        # 读取光流数据
        flow_data = np.fromfile(f, np.float32, count=2 * width * height)
        flow_data = flow_data.reshape((height, width, 2))
        
        # 打印结果
        print("Width:", width)
        print("Height:", height)
    
    return flow_data



def calculate_feature_map(img_t, flow_O1):
    # 获取图像和光流的形状
    height, width = img_t.shape
    feature_map = np.zeros_like(img_t, dtype=np.float32)

    for y in range(height):
        for x in range(width):
            y = min(y, flow_O1.shape[0] - 1)  # 高度限制
            x = min(x, flow_O1.shape[1] - 1)  # 宽度限制
            # 获取光流矢量
            flow_vector = flow_O1[y, x]  # [dy, dx]
            # 计算 t-1 时刻图像的像素位置
            t_minus_1_x = int(x - flow_vector[0])
            t_minus_1_y = int(y - flow_vector[1])

            # 确保坐标在图像范围内
            if 0 <= t_minus_1_x < width and 0 <= t_minus_1_y < height:
                feature_map[y, x] = img_t[y, x] - img_t[t_minus_1_y, t_minus_1_x]
    
    return feature_map

def calculate_feature_map_(img_t, flow_O1):
    # 获取图像和光流的形状
    flow_resized = cv2.resize(flow_O1, (img_t.shape[0], img_t.shape[1]), interpolation=cv2.INTER_LINEAR)
    height, width = img_t.shape
    feature_map = np.zeros_like(img_t, dtype=np.float32)

    for y in range(height):
        for x in range(width):
            y = min(y, flow_O1.shape[0] - 1)  # 高度限制
            x = min(x, flow_O1.shape[1] - 1)  # 宽度限制
            # 获取光流矢量
            flow_vector = flow_O1[y, x]  # [dy, dx]
            # 计算 t-1 时刻图像的像素位置
            t_minus_1_x = int(x + flow_vector[0])
            t_minus_1_y = int(y + flow_vector[1])

            # 确保坐标在图像范围内
            if 0 <= t_minus_1_x < width and 0 <= t_minus_1_y < height:
                feature_map[y, x] = img_t[y, x] - img_t[t_minus_1_y, t_minus_1_x]
    
    return feature_map


# 文件夹路径
image_folder = "/root/autodl-tmp/dataset1-70/train/input"
flow_folder = "/root/autodl-tmp/dataset1-70/result/inference/run.epoch-0-flow-field"
save_folder = "/root/autodl-tmp/dataset1-70/result/feature_maps/"

# 获取所有 PNG 和 FLO 文件的文件名
image_files = sorted([f for f in os.listdir(image_folder) if f.endswith('.png')])
flow_files = sorted([f for f in os.listdir(flow_folder) if f.endswith('.flo')])

# 确保保存特征图的文件夹存在
if not os.path.exists(save_folder):
    os.makedirs(save_folder)

image_list = list()

def collect_images(image_folderr):
    # 获取文件夹中所有的png文件
    sorted_png = sorted([f for f in os.listdir(image_folder) if f.endswith('.png')])
    # 遍历所有文件，寻找相邻图像对
    image_root = image_folder
    for i in range(len(sorted_png)-2):
        file1 = sorted_png[i]
        file2 = sorted_png[i+1]
        file3 = sorted_png[i+2]
        
        # 使用正则表达式匹配文件名中的 P 和 slice 以及 time
        match1 = re.match(r'(lax|sax)_P(\d+)_slice_(\d+)_time_(\d+)\.png', file1)
        match2 = re.match(r'(lax|sax)_P(\d+)_slice_(\d+)_time_(\d+)\.png', file2)
        match3 = re.match(r'(lax|sax)_P(\d+)_slice_(\d+)_time_(\d+)\.png', file3)

        # 确保两个文件名都匹配模式
        if match1 and match2 and match3:
            ls_1 = match1.group(1)
            ls_2 = match2.group(1)
            ls_3 = match3.group(1)
            P_1 = int(match1.group(2))
            P_2 = int(match2.group(2))
            P_3 = int(match3.group(2))
            slice_num1 = int(match1.group(3))
            slice_num2 = int(match2.group(3))
            slice_num3 = int(match3.group(3))
            # 检查P值和slice是否相同
            if slice_num1 == slice_num2 and P_1 == P_2 and P_1 == P_3 and slice_num1 == slice_num3 and ls_3 == ls_1 and ls_2 == ls_3:
                # 获取 time 编号
                fnum1 = int(match1.group(4))
                fnum2 = int(match2.group(4))
                fnum3 = int(match3.group(4))
                
                # 打印slice和time编号调试用
                print(f"L/S: {ls_1} P: {P_1}, Slice: {slice_num1}, Time1: {fnum1}, Time2: {fnum2},Time3: {fnum3}")
                
                # 创建图像1和图像2的路径
                img1 = join(image_root, file1)  # 完整的png文件路径
                img2 = join(image_root, file2)
                img3 = join(image_root, file3)
                
                # 将图像对加入到列表中
                image_list.append([img1, img2, img3])
    return image_list
                
image_list = collect_images(image_folder)
print(len(image_list))
for i in tqdm(range(len(image_list)-2)):
    img_t_minus_1_path = os.path.join(image_folder, image_list[i][0])
    img_t_path = os.path.join(image_folder, image_list[i][1])
    img_t_plus_1_path = os.path.join(image_folder, image_list[i][2])

    # 去掉扩展名
    file_name_with_ext = os.path.basename(img_t_minus_1_path)
    file_name_0 = os.path.splitext(file_name_with_ext)[0]
    file_name_with_ext = os.path.basename(img_t_plus_1_path)
    file_name_2 = os.path.splitext(file_name_with_ext)[0]
    
    
    # 读取图像
    img_t_minus_1 = cv2.imread(img_t_minus_1_path, cv2.IMREAD_GRAYSCALE)
    img_t = cv2.imread(img_t_path, cv2.IMREAD_GRAYSCALE)
    img_t_plus_1 = cv2.imread(img_t_plus_1_path, cv2.IMREAD_GRAYSCALE)
    
    flow_O1_path = os.path.join(flow_folder, flow_files[i])
    flow_O2_path = os.path.join(flow_folder, flow_files[i+1])
    flow_O1 = read_flo_file(flow_O1_path)
    flow_O2 = read_flo_file(flow_O2_path)
    
    feature_map_t_minus_1 = calculate_feature_map(img_t, flow_O1)
    feature_map_t_plus_1 = calculate_feature_map_(img_t, flow_O2)
        # 转换为可视化格式并保存
    feature_map_visual_t_minus_1 = cv2.normalize(feature_map_t_minus_1, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    feature_map_visual_t_plus_1 = cv2.normalize(feature_map_t_plus_1, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    # 生成保存路径
    feature_map_t_minus_1_path = os.path.join(save_folder, f"{file_name_0}_0.png")
    feature_map_t_plus_1_path = os.path.join(save_folder, f"{file_name_2}_1.png")
    
    # 保存特征图
    cv2.imwrite(feature_map_t_minus_1_path, feature_map_visual_t_minus_1)
    cv2.imwrite(feature_map_t_plus_1_path, feature_map_visual_t_plus_1)


print("Feature maps have been saved.")