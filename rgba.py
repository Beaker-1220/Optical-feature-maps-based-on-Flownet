import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.models as models

# 加载图片
image_path = '/root/autodl-tmp/train_png/lax_P001_slice_00_time_00.png'  # 替换为你的图片路径
rgba_image = Image.open(image_path)

# 分离RGB和Alpha通道
r, g, b, a = rgba_image.split()

# 合并RGB
rgb_image = Image.merge('RGB', (r, g, b))

# 显示RGB和Alpha通道
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# 显示RGB图像
axes[0].imshow(rgb_image)
axes[0].set_title('RGB Image')
axes[0].axis('off')

# 显示Alpha通道
axes[1].imshow(a, cmap='gray')
axes[1].set_title('Alpha Channel')
axes[1].axis('off')

plt.tight_layout()
output_image_path = '/root/result/feature_maps/rgba.png'  # 替换为你想要的保存路径
plt.savefig(output_image_path, bbox_inches='tight')