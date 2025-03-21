import torch
import torch.utils.data as data
import os
import re
import os, math, random
from os.path import *
import numpy as np
import torchvision.transforms as transforms  # 导入 transforms 模块
from glob import glob
import utils.frame_utils as frame_utils
from PIL import Image
from imageio import imread

class StaticRandomCrop(object):
    def __init__(self, image_size, crop_size):
        self.th, self.tw = crop_size
        h, w = image_size
        self.h1 = random.randint(0, h - self.th)
        self.w1 = random.randint(0, w - self.tw)

    def __call__(self, img):
        return img[self.h1:(self.h1+self.th), self.w1:(self.w1+self.tw),:]

class StaticCenterCrop(object):
    def __init__(self, image_size, crop_size):
        self.th, self.tw = crop_size
        self.h, self.w = image_size
    def __call__(self, img):
        return img[(self.h-self.th)//2:(self.h+self.th)//2, (self.w-self.tw)//2:(self.w+self.tw)//2,:]

class MpiSintel(data.Dataset):
    def __init__(self, args, is_cropped = False, root = '', dstype = 'clean', replicates = 1):
        self.args = args
        self.is_cropped = is_cropped
        self.crop_size = args.crop_size
        self.render_size = args.inference_size
        self.replicates = replicates

        flow_root = join(root)
        image_root = join(root)
        print(flow_root)
        print(image_root)
        def sort_ppm_files(folder_path):
            # 获取所有png文件
            files = [f for f in os.listdir(folder_path) if f.endswith('.png')]
                # Helper function to extract P, slice, and time from the filename

            # 定义排序的关键函数
            def sorting_key(filename):
                match = re.match(r'lax_(P\d+)_slice_(\d+)_time_(\d+).png', filename)
                if match:
                    P, slice_num, time = match.groups()
                    return (P, slice_num, time_num)  # 返回一个元组作为排序关键字
                return (0, 0, 0)  # 如果不匹配，则放在最前面

            # 按照slice和time排序
            sorted_files = sorted(files, key=sorting_key)

            return sorted_files

        # 使用示例
        folder_path = image_root  # 替换为你的文件夹路径
        sorted_ppm_files = sort_ppm_files(folder_path)
        # for file in sorted_ppm_files:
        #     print(file)
            
        self.flow_list = []
        self.image_list = []

        for i in range(len(sorted_ppm_files) ):
            if 'test' in file:
                # print file
                continue
            file1 = sorted_ppm_files[i-1]
            file2 = sorted_ppm_files[i]
            print(file1[:-4])
                    # 提取slice编号
            match1 = re.match(r'lax_(P\d+)_slice_(\d+)_time_(\d+)\.png', file1)
            match2 = re.match(r'lax_(P\d+)_slice_(\d+)_time_(\d+)\.png', file2)

            if match1 and match2:
                P_1 = int(match1.group(1))
                P_2 = int(match2.group(2))
                slice_num1 = int(match1.group(2))
                slice_num2 = int(match2.group(2))
                print(slice_num1)

                # 如果slice相同，设置img1和img2
                if slice_num1 == slice_num2 and P_1 == P_2:
                    # 获取文件编号
                    fnum1 = int(match1.group(2))  # 从filename中提取time编号
                    fnum2 = int(match2.group(2))  # 从filename中提取time编号
                    print(fnum1)
                    
                    # 创建img1和img2的路径
                    img1 = join(image_root, f"{file1[:-4]}.png")  # 假设是png格式
                    img2 = join(image_root, f"{file2[:-4]}.png")  # 假设是png格式
                    

                    self.image_list += [[img1, img2]] # 添加到图片列表中
                    self.flow_list += [file]

        self.size = len(self.image_list)
        print(self.size)
        self.frame_size = frame_utils.read_gen(self.image_list[0][0]).shape

        if (self.render_size[0] < 0) or (self.render_size[1] < 0) or (self.frame_size[0]%64) or (self.frame_size[1]%64):
            self.render_size[0] = ( (self.frame_size[0])//64 ) * 64
            self.render_size[1] = ( (self.frame_size[1])//64 ) * 64

        args.inference_size = self.render_size

        assert (len(self.image_list) == len(self.flow_list))
    
        # 定义排序的关键函数


    def __getitem__(self, index):

        index = index % self.size
        img1 = frame_utils.read_gen(self.image_list[index][0])
        img2 = frame_utils.read_gen(self.image_list[index][1])

        flow = frame_utils.read_gen(self.flow_list[index])

        images = [img1, img2]
        image_size = img1.shape[:2]

        if self.is_cropped:
            cropper = StaticRandomCrop(image_size, self.crop_size)
        else:
            cropper = StaticCenterCrop(image_size, self.render_size)
        images = list(map(cropper, images))
        flow = cropper(flow)

        images = np.array(images).transpose(3,0,1,2)
        flow = flow.transpose(2,0,1)

        images = torch.from_numpy(images.astype(np.float32))
        flow = torch.from_numpy(flow.astype(np.float32))

        return [images], [flow]

    def __len__(self):
        return self.size * self.replicates

class MpiSintelClean(MpiSintel):
    def __init__(self, args, is_cropped = False, root = '', replicates = 1):
        super(MpiSintelClean, self).__init__(args, is_cropped = is_cropped, root = root, dstype = 'clean', replicates = replicates)

class MpiSintelFinal(MpiSintel):
    def __init__(self, args, is_cropped = False, root = '', replicates = 1):
        super(MpiSintelFinal, self).__init__(args, is_cropped = is_cropped, root = root, dstype = 'final', replicates = replicates)

class FlyingChairs(data.Dataset):
  def __init__(self, args, is_cropped, root = '/path/to/FlyingChairs_release/data', replicates = 1):
    self.args = args
    self.is_cropped = is_cropped
    self.crop_size = args.crop_size
    self.render_size = args.inference_size
    self.replicates = replicates

    images = sorted( glob( join(root, '*.ppm') ) )

    self.flow_list = sorted( glob( join(root, '*.flo') ) )

    assert (len(images)//2 == len(self.flow_list))

    self.image_list = []
    for i in range(len(self.flow_list)):
        im1 = images[2*i]
        im2 = images[2*i + 1]
        self.image_list += [ [ im1, im2 ] ]

    assert len(self.image_list) == len(self.flow_list)

    self.size = len(self.image_list)

    self.frame_size = frame_utils.read_gen(self.image_list[0][0]).shape

    if (self.render_size[0] < 0) or (self.render_size[1] < 0) or (self.frame_size[0]%64) or (self.frame_size[1]%64):
        self.render_size[0] = ( (self.frame_size[0])//64 ) * 64
        self.render_size[1] = ( (self.frame_size[1])//64 ) * 64

    args.inference_size = self.render_size

  def __getitem__(self, index):
    index = index % self.size

    img1 = frame_utils.read_gen(self.image_list[index][0])
    img2 = frame_utils.read_gen(self.image_list[index][1])

    flow = frame_utils.read_gen(self.flow_list[index])

    images = [img1, img2]
    image_size = img1.shape[:2]
    if self.is_cropped:
        cropper = StaticRandomCrop(image_size, self.crop_size)
    else:
        cropper = StaticCenterCrop(image_size, self.render_size)
    images = list(map(cropper, images))
    flow = cropper(flow)


    images = np.array(images).transpose(3,0,1,2)
    flow = flow.transpose(2,0,1)

    images = torch.from_numpy(images.astype(np.float32))
    flow = torch.from_numpy(flow.astype(np.float32))

    return [images], [flow]

  def __len__(self):
    return self.size * self.replicates

class FlyingThings(data.Dataset):
  def __init__(self, args, is_cropped, root = '/path/to/flyingthings3d', dstype = 'frames_cleanpass', replicates = 1):
    self.args = args
    self.is_cropped = is_cropped
    self.crop_size = args.crop_size
    self.render_size = args.inference_size
    self.replicates = replicates

    image_dirs = sorted(glob(join(root, dstype, 'TRAIN/*/*')))
    image_dirs = sorted([join(f, 'left') for f in image_dirs] + [join(f, 'right') for f in image_dirs])

    flow_dirs = sorted(glob(join(root, 'optical_flow_flo_format/TRAIN/*/*')))
    flow_dirs = sorted([join(f, 'into_future/left') for f in flow_dirs] + [join(f, 'into_future/right') for f in flow_dirs])

    assert (len(image_dirs) == len(flow_dirs))

    self.image_list = []
    self.flow_list = []

    for idir, fdir in zip(image_dirs, flow_dirs):
        images = sorted( glob(join(idir, '*.png')) )
        flows = sorted( glob(join(fdir, '*.flo')) )
        for i in range(len(flows)):
            self.image_list += [ [ images[i], images[i+1] ] ]
            self.flow_list += [flows[i]]

    assert len(self.image_list) == len(self.flow_list)

    self.size = len(self.image_list)

    self.frame_size = frame_utils.read_gen(self.image_list[0][0]).shape

    if (self.render_size[0] < 0) or (self.render_size[1] < 0) or (self.frame_size[0]%64) or (self.frame_size[1]%64):
        self.render_size[0] = ( (self.frame_size[0])//64 ) * 64
        self.render_size[1] = ( (self.frame_size[1])//64 ) * 64

    args.inference_size = self.render_size

  def __getitem__(self, index):
    index = index % self.size

    img1 = frame_utils.read_gen(self.image_list[index][0])
    img2 = frame_utils.read_gen(self.image_list[index][1])

    flow = frame_utils.read_gen(self.flow_list[index])

    images = [img1, img2]
    image_size = img1.shape[:2]
    if self.is_cropped:
        cropper = StaticRandomCrop(image_size, self.crop_size)
    else:
        cropper = StaticCenterCrop(image_size, self.render_size)
    images = list(map(cropper, images))
    flow = cropper(flow)


    images = np.array(images).transpose(3,0,1,2)
    flow = flow.transpose(2,0,1)

    images = torch.from_numpy(images.astype(np.float32))
    flow = torch.from_numpy(flow.astype(np.float32))

    return [images], [flow]

  def __len__(self):
    return self.size * self.replicates

class FlyingThingsClean(FlyingThings):
    def __init__(self, args, is_cropped = False, root = '', replicates = 1):
        super(FlyingThingsClean, self).__init__(args, is_cropped = is_cropped, root = root, dstype = 'frames_cleanpass', replicates = replicates)

class FlyingThingsFinal(FlyingThings):
    def __init__(self, args, is_cropped = False, root = '', replicates = 1):
        super(FlyingThingsFinal, self).__init__(args, is_cropped = is_cropped, root = root, dstype = 'frames_finalpass', replicates = replicates)

class ChairsSDHom(data.Dataset):
  def __init__(self, args, is_cropped, root = '/path/to/chairssdhom/data', dstype = 'train', replicates = 1):
    self.args = args
    self.is_cropped = is_cropped
    self.crop_size = args.crop_size
    self.render_size = args.inference_size
    self.replicates = replicates

    image1 = sorted( glob( join(root, dstype, 't0/*.png') ) )
    image2 = sorted( glob( join(root, dstype, 't1/*.png') ) )
    self.flow_list = sorted( glob( join(root, dstype, 'flow/*.flo') ) )

    assert (len(image1) == len(self.flow_list))

    self.image_list = []
    for i in range(len(self.flow_list)):
        im1 = image1[i]
        im2 = image2[i]
        self.image_list += [ [ im1, im2 ] ]

    assert len(self.image_list) == len(self.flow_list)

    self.size = len(self.image_list)

    self.frame_size = frame_utils.read_gen(self.image_list[0][0]).shape

    if (self.render_size[0] < 0) or (self.render_size[1] < 0) or (self.frame_size[0]%64) or (self.frame_size[1]%64):
        self.render_size[0] = ( (self.frame_size[0])//64 ) * 64
        self.render_size[1] = ( (self.frame_size[1])//64 ) * 64

    args.inference_size = self.render_size

  def __getitem__(self, index):
    index = index % self.size

    img1 = frame_utils.read_gen(self.image_list[index][0])
    img2 = frame_utils.read_gen(self.image_list[index][1])

    flow = frame_utils.read_gen(self.flow_list[index])
    flow = flow[::-1,:,:]

    images = [img1, img2]
    image_size = img1.shape[:2]
    if self.is_cropped:
        cropper = StaticRandomCrop(image_size, self.crop_size)
    else:
        cropper = StaticCenterCrop(image_size, self.render_size)
    images = list(map(cropper, images))
    flow = cropper(flow)


    images = np.array(images).transpose(3,0,1,2)
    flow = flow.transpose(2,0,1)

    images = torch.from_numpy(images.astype(np.float32))
    flow = torch.from_numpy(flow.astype(np.float32))

    return [images], [flow]

  def __len__(self):
    return self.size * self.replicates

class ChairsSDHomTrain(ChairsSDHom):
    def __init__(self, args, is_cropped = False, root = '', replicates = 1):
        super(ChairsSDHomTrain, self).__init__(args, is_cropped = is_cropped, root = root, dstype = 'train', replicates = replicates)

class ChairsSDHomTest(ChairsSDHom):
    def __init__(self, args, is_cropped = False, root = '', replicates = 1):
        super(ChairsSDHomTest, self).__init__(args, is_cropped = is_cropped, root = root, dstype = 'test', replicates = replicates)

class ImagesFromFolder(data.Dataset):
  def __init__(self, args, is_cropped, root = '/path/to/frames/only/folder', iext = 'png', replicates = 1):
    self.args = args
    self.is_cropped = is_cropped
    self.crop_size = args.crop_size
    self.render_size = args.inference_size
    self.replicates = replicates
    image_root = root
    
    def sort_ppm_files(folder_path):
                # 获取所有png文件
                files = [f for f in os.listdir(folder_path) if f.endswith('.png')]
                    # Helper function to extract P, slice, and time from the filename

                # 定义排序的关键函数
                def sorting_key(filename):
                    match = re.match(r'(lax|sax)_(P\d+)_slice_(\d+)_time_(\d+).png', filename)
                    if match:
                        prefix, P, slice_num, time = match.groups()
                        return (prefix, P, int(slice_num), int(time))  # 返回一个元组作为排序关键字
                    return (0, 0, 0, 0)  # 如果不匹配，则放在最前面

                # 按照slice和time排序
                sorted_files = sorted(files, key=sorting_key)

                return sorted_files

            # 使用示例
    folder_path = root  # 替换为你的文件夹路径
    sorted_png = sort_ppm_files(folder_path)
    # for f in sorted_png:
    #     print(f)

    self.image_list = []
    for i in range(len(sorted_png)-1):
        file1 = sorted_png[i]
        file2 = sorted_png[i+1]
        match1 = re.match(r'(lax|sax)_P(\d+)_slice_(\d+)_time_(\d+)\.png', file1)
        match2 = re.match(r'(lax|sax)_P(\d+)_slice_(\d+)_time_(\d+)\.png', file2)

        if match1 and match2:
            ls_1 = match1.group(1)
            ls_2 = match2.group(1)
            print(ls_1, ls_2)
            P_1 = int(match1.group(2))
            P_2 = int(match2.group(2))
            slice_num1 = int(match1.group(3))
            slice_num2 = int(match2.group(3))


            # 如果slice相同，设置img1和img2
            if slice_num1 == slice_num2 and P_1 == P_2 and ls_1 == ls_2:
                # 获取文件编号
                fnum1 = int(match1.group(4))  # 从filename中提取time编号
                fnum2 = int(match2.group(4))  # 从filename中提取time编号

                
                # 创建img1和img2的路径
                img1 = join(image_root, f"{file1[:-4]}.png")  # 假设是png格式
                img2 = join(image_root, f"{file2[:-4]}.png")  # 假设是png格式a


                self.image_list += [ [ img1, img2 ] ]

    self.size = len(self.image_list)
    print("Image List:", self.image_list)
    print('size', self.size)



    self.frame_size = frame_utils.read_gen(self.image_list[0][0]).shape

    if (self.render_size[0] < 0) or (self.render_size[1] < 0) or (self.frame_size[0]%64) or (self.frame_size[1]%64):
        self.render_size[0] = ( (self.frame_size[0])//64 ) * 64
        self.render_size[1] = ( (self.frame_size[1])//64 ) * 64

    args.inference_size = self.render_size

  def __getitem__(self, index):
    index = index % self.size

    img1 = frame_utils.read_gen(self.image_list[index][0])
    img2 = frame_utils.read_gen(self.image_list[index][1])



    images = [img1, img2]
    image_size = img1.shape[:2]
    if self.is_cropped:
        cropper = StaticRandomCrop(image_size, self.crop_size)
    else:
        cropper = StaticCenterCrop(image_size, self.render_size)
    images = list(map(cropper, images))
    # 将 NumPy 数组转换为 PIL 图像
    images = [Image.fromarray(img) for img in images]
    
    # 创建 Resize 操作，将图像大小固定为 204x448
    resize_transform = transforms.Resize((448, 192))
    
    # 调整图像大小
    images = [resize_transform(img) for img in images]
    
    # 将 PIL 图像转换回 NumPy 数组并进行维度变换
    images = [np.array(img) for img in images]
    images = np.array(images).transpose(3, 0, 1, 2)
    
    # 将 NumPy 数组转换为 PyTorch 张量
    images = torch.from_numpy(images.astype(np.float32))
    return [images], [torch.zeros(images.size()[0:1] + (2,) + images.size()[-2:])]
  def __len__(self):
    return self.size * self.replicates
'''
import argparse
import sys, os
import importlib
from scipy.misc import imsave
import numpy as np

import datasets
reload(datasets)

parser = argparse.ArgumentParser()
args = parser.parse_args()
args.inference_size = [1080, 1920]
args.crop_size = [384, 512]
args.effective_batch_size = 1

index = 500
v_dataset = datasets.MpiSintelClean(args, True, root='../MPI-Sintel/flow/training')
a, b = v_dataset[index]
im1 = a[0].numpy()[:,0,:,:].transpose(1,2,0)
im2 = a[0].numpy()[:,1,:,:].transpose(1,2,0)
imsave('./img1.png', im1)
imsave('./img2.png', im2)
flow_utils.writeFlow('./flow.flo', b[0].numpy().transpose(1,2,0))

'''
