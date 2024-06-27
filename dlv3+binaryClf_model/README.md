# II. Binary Classification using DeeplabV3+

We have used Segmentation Backbone of DeepLabv3+ model pre-trained on eMARG-15k(Good/Bad) and extended it for Binary Classification by adding simple Conv + FC layer combination layers. 

## Quick Start 
## Architecture of DeeplabV3+ Fine-tuned for Binary Classification.
<p float="left">
 <img src="BinaryClf_model/binary_clf.png" alt="Image" width="600" />
  
</p>


| DeepLabV3    |  DeepLabV3+        |
| :---: | :---:     |
|deeplabv3_resnet50|deeplabv3plus_resnet50|
|deeplabv3_resnet101|deeplabv3plus_resnet101|
|deeplabv3_mobilenet|deeplabv3plus_mobilenet ||
|deeplabv3_hrnetv2_48 | deeplabv3plus_hrnetv2_48 |
|deeplabv3_hrnetv2_32 | deeplabv3plus_hrnetv2_32 |

### All pretrained model checkpoints: [Drive](https://drive.google.com/drive/folders/1jA0iS7hq-AmFBtSn0Ne9DvIxcXCQBQGG?usp=drive_link)

### 1. Load the pretrained model:
```python
model.load_state_dict( torch.load( CKPT_PATH )['model_state']  )
```


### 2. Prediction
Single image:
```bash
python predict.py --input datasets/data/eMARG/leftImg8bit/train/city0/PE-AR-7382-157_2_leftImg8bit  --dataset cityscapes --model deeplabv3plus_mobilenet --ckpt checkpoints/best_deeplabv3plus_mobilenet_cityscapes_os16.pth --save_val_results_to test_results
```

Image folder:
```bash
python predict.py --input datasets/data/eMARG/leftImg8bit/train/city0  --dataset cityscapes --model deeplabv3plus_mobilenet --ckpt checkpoints/best_deeplabv3plus_mobilenet_cityscapes_os16.pth --save_val_results_to test_results
```


## Results

### 1. Performance on eMARG (6 classes, 512 x 384)

Training: 768x768 random crop  
validation: 512x384

|  Model          | Batch Size  | Accuracy  | Precision   |  Recall  |F1-score |checkpoint_link   |
| :--------        | :-------------: | :----:   | :-----------: | :--------: | :--------: |  :----:   |
| DeepLabV3Plus-ResNet101    | 4    | 0.884     |  0.8618 |  0.915  |  0.887 | [Download](https://drive.google.com/file/d/1G5hRKOnwDCcLVnX-Sgnxz4bzsndWNxKV/view?usp=drive_link)
| DeepLabV3Plus-MobileNet  |   8   |  0.869   |  0.841   |  0.908  | 0.874 | [Download](https://drive.google.com/file/d/1G5hRKOnwDCcLVnX-Sgnxz4bzsndWNxKV/view?usp=drive_link)

## GradCAM Results on eMARG (DeepLabv3Plus-MobileNet/ResNet-101)

<p float="left">
  <img src="gradcam_results/PE-GJ-84921-20_2.jpg" alt="Image" width="300" />
  <img src="gradcam_results/PE-GJ-84921-20_2_overlay.jpg" width="300" /> 
  <img src="gradcam_results/PE-GJ-85702-59_2.jpg" width="300" />
  <img src="gradcam_results/PE-GJ-85702-59_2_overlay.jpg" width="300" />
  
</p>



## eMARG Dataset
### 1. Requirements

```bash
pip install -r requirements.txt
```


### 2. Download eMARG and extract it likewise Cityscapes dataset in this format 'datasets/data/eMARG'

```
/datasets
    /data
        /eMARG
            /gtFine
            /leftImg8bit
```

### 3. Train your model on eMARG likewise Cityscapes.

```bash
python main.py --model deeplabv3plus_mobilenet --dataset cityscapes --enable_vis --vis_port 28333 --gpu_id 0  --lr 0.1  --crop_size 768 --batch_size 16 --output_stride 16 --data_root ./datasets/data/eMARG
python main.py --model deeplabv3plus_resnet101 --dataset cityscapes --enable_vis --vis_port 28333 --gpu_id 0  --lr 0.1  --crop_size 768 --batch_size 16 --output_stride 16 --data_root ./datasets/data/eMARG 
```

#### 4. Testing

Results will be saved at ./results.

```bash
python main.py --model deeplabv3plus_mobilenet --enable_vis --vis_port 28333 --gpu_id 0 --year 2012_aug --crop_val --lr 0.01 --crop_size 513 --batch_size 16 --output_stride 16 --ckpt checkpoints/best_deeplabv3plus_mobilenet_cityscapes_os16.pth --test_only --save_val_results
python main.py --model deeplabv3plus_mobilenet --enable_vis --vis_port 28333 --gpu_id 0 --year 2012_aug --crop_val --lr 0.01 --crop_size 513 --batch_size 16 --output_stride 16 --ckpt checkpoints/best_deeplabv3plus_mobilenet_cityscapes_os16.pth --test_only --save_val_results

```

## Reference

[1] [Rethinking Atrous Convolution for Semantic Image Segmentation](https://arxiv.org/abs/1706.05587)

[2] [Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation](https://arxiv.org/abs/1802.02611)

