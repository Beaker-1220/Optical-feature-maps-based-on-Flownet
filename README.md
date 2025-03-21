# Optical-feature-maps-based-on-Flownet
# flownet
## optical inference

    python main.py --inference --model FlowNet2 \
    --save_flow --inference_dataset ImagesFromFolder \
    --resume /root/FlowNet2_checkpoint.pth.tar \
    --save /root/autodl-tmp/result \
    --inference_dataset_root /root/autodl-tmp/train_png \
    --number_gpus 1

## Get optcial featuremaps
    python fm.py
And you can adjust the dataset path and Name of the file in the fm.py

## What it does?

Flownet extracts the vector information file.flo of the pixel displacement from consecutive image frames. The fm.py designs an algorithm that takes three consecutive frames as a group: t-1, t, and t+1. By adding the vector displacement predicted by the.flo file to the image at t-1, an optical feature map of the image at time t is obtained. The same operation is carried out for the image at time t+1. Eventually, two optical feature maps for each image are obtained, which are used to incorporate the temporal and motion information of consecutive frames later, so as to enhance the effect of MRI reconstruction.
