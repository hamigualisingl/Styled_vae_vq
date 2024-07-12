
import torch
import json
import os

from PIL import Image
# from models.vqvae import VQVAE
from model import VQVAE_Transformer_vit_sd3_hug_4096
from torchvision.transforms import InterpolationMode, transforms
from torchvision import transforms
def normalize_01_into_pm1(x):  # normalize x from [0, 1] to [-1, 1] by (x*2) - 1
    return x.add(x).add_(-1)
def get_state_dict(model_weight):
    state_dict = torch.load(model_weight)
    state_dict_removed = {}
    for k, value in state_dict.items():
        if "module." in k:
            k_removed = k.split("module.")[-1]
            state_dict_removed[k_removed] = value
        else:
            state_dict_removed[k] = value
    return state_dict_removed
val_aug = transforms.Compose([
   transforms.Resize((256,256), interpolation=InterpolationMode.LANCZOS),
   
        transforms.ToTensor(), normalize_01_into_pm1,   # 将PIL Image或者numpy.ndarray转换为Tensor
    # 应用自定义的归一化到[-1, 1]
])
device =  'cuda'
epoch_number = 97
file_path = "/mnt/data/user/lidehu/vae/ALIP/out_put_stage1_6expert_std_noise_0.1_pect_1/model_{:d}.pt".format(epoch_number)
model_alip =VQVAE_Transformer_vit_sd3_hug_4096(width=1024, layers=24, heads=16, mlp_ratio=4.0)
state_dict = get_state_dict(file_path)
model_alip.load_state_dict(state_dict, strict=True)
# 确保在推理前模型处于评估模式
model_alip.to(device)  
model_alip.eval()
image_path = ['/mnt/data/user/lidehu/VAR/WechatIMG37.jpg','/mnt/data/user/lidehu/VAR/resized_image.jpg','/mnt/data/user/lidehu/VAR/image.jpg','/mnt/data/user/lidehu/VAR/20240527-3.jpg',"/mnt/data/user/lidehu/VAR/20240527-4.jpg","/mnt/data/user/lidehu/VAR/20240527-2.jpg"]  # 替换为你的图片路径

for i in range(len(image_path)):
    fin=i
    image = Image.open(image_path[fin]).convert('RGB')
    original_output_path = f"/mnt/data/user/lidehu/vae/ALIP/aul/how_original_image_{fin}.jpg"
    image = image.resize((256, 256), Image.ANTIALIAS)  # ANTIALIAS用于平滑缩放
    image.save(original_output_path, format='JPEG')
    print(f"Original image saved at {original_output_path}")
# 应用变换
    img = val_aug(image).unsqueeze(0).to(device) 
    
    reconstructed_images=model_alip.infer(img)#(260, 4, 96) 因为只取的最后的
    img=reconstructed_images.detach().add_(1).mul_(0.5)[0].squeeze(0)
                                               #因为是-1 到1 ，加1就是0-2 然后0-1                      #修改这里，可以选择保存哪张图片
#img=vae.idxBl_to_img(idxbl,True,last_one=True).add_(1).mul_(0.5).squeeze(0)
    img_array = img.permute(1, 2, 0).cpu().numpy() * 255  # 转置通道顺序并乘以255使其成为0-255范围内的像素值
# 将NumPy数组转换为PIL图像
    img_pil = Image.fromarray(img_array.astype('uint8'))  # 注意astype转换为无符号整型
    output_path = f"/mnt/data/user/lidehu/vae/ALIP/aul/styled{epoch_number}_{fin}.jpg"
    img_pil.save(output_path, format='JPEG')
    print(f"Reconstructed image saved at {output_path}")
