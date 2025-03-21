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
