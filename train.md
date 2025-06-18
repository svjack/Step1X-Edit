### dataset
```python
from datasets import load_dataset 
from tqdm import tqdm
import os

# Create directory (note: the correct parameter is exist_ok, not exists_ok)
os.makedirs("poster_dir", exist_ok=True)

ds = load_dataset("svjack/poster_text_render")["train"]

# Sort the dataset by some key if needed (you might want to add a sorting key)
# ds = sorted(ds, key=lambda x: x['some_key']) 

for idx, ele in enumerate(tqdm(ds)):
    # Resize images
    poster_img = ele["poster_image"].resize([768, 1024])
    inpaint_img = ele["inpaint_image"].resize([768, 1024])
    
    # Save images with sequential naming
    poster_img.save(f"poster_dir/poster_{idx:04d}.jpg")  # :04d pads with zeros to 4 digits
    inpaint_img.save(f"poster_dir/inpaint_{idx:04d}.jpg")
```

```python
import pandas as pd 
import pathlib 
l0 = pd.Series(list(pathlib.Path("poster_dir").rglob("poster_*.jpg"))).map(str).sort_values().values.tolist()
l1 = pd.Series(list(pathlib.Path("poster_dir").rglob("inpaint_*.jpg"))).map(str).sort_values().values.tolist()

from functools import reduce

d = dict(
reduce(lambda a, b: a + b,
pd.DataFrame(list(zip(*(l0, l1)))).applymap(lambda x: os.path.join("/home/featurize/Step1X-Edit", x)).applymap(
    lambda x: x
).apply(
    lambda x: 
        {
            x[1]:{
                    'ref_image_path': x[0],
                    'caption': "remove the text from the image"
            }
        }
    , axis = 1
).map(lambda d: list(d.items())).values.tolist()))
len(d)
meta_data_path = "rm_meta.json"
import json 
with open(meta_data_path, "w") as f:
    json.dump(d, f)
```

### code edit
```bash
mode="flash"
in 
Step1X-Edit/library/step1x_utils.py 

step1x_params = Step1XParams
```

### train

```toml
[general]
shuffle_caption = false
caption_extension = '.txt'
keep_tokens = 1

# This is a edit dataset
[[datasets]]
resolution = [768, 1024]
batch_size = 1
edit_dataset = true # necessary for editing tasks

[[datasets.subsets]]
image_dir = "poster_dir"
metadata_file = "rm_meta.json"
```

```bash
accelerate launch  --mixed_precision bf16 --num_cpu_threads_per_process 1 --num_processes 1 \
--config_file ./library/accelerate_config.yaml \
finetuning.py \
--pretrained_model_name_or_path step1x-edit-i1258.safetensors \
--qwen2p5vl Qwen2.5-VL-7B-Instruct \
--ae vae.safetensors \
--cache_latents_to_disk --save_model_as safetensors --sdpa --persistent_data_loader_workers \
--max_data_loader_n_workers 2 --seed 42 --gradient_checkpointing --mixed_precision bf16 --save_precision bf16 \
--network_module library.lora_module --network_dim 64 --network_alpha 32 --network_train_unet_only \
--optimizer_type adamw8bit --learning_rate 1e-4 \
--cache_text_encoder_outputs --cache_text_encoder_outputs_to_disk \
--highvram --max_train_epochs 100 --save_every_n_epochs 5 --dataset_config step1x_edit.toml \
--output_dir rm_text_output \
--output_name step1x-edit_rm_text \
--timestep_sampling shift --discrete_flow_shift 3.1582 --model_prediction_type raw --guidance_scale 1.0 --fp8_base
```

### inference
```python
from inference import *

image_edit = ImageGenerator(
        ae_path="vae.safetensors",
        dit_path="step1x-edit-i1258.safetensors",
        qwen2vl_model_path='Qwen2.5-VL-7B-Instruct',
        max_length=640,
        quantized=True,
        offload=True,
        lora="rm_text_output/step1x-edit_rm_text-000025.safetensors",
        mode="flash"
    )

import pandas as pd
import pathlib 
from tqdm import tqdm
import os
os.makedirs("benchmark", exist_ok= True)
l = pd.Series(list(pathlib.Path("../OWLSAM/").rglob("*original*.png"))).map(str).values.tolist()
for image_path in tqdm(l):
    num_steps = 28
    cfg_guidance = 4.5
    seed  = 42
    size_level = 512
    #size_level = 768
    #size_level = 1024
    image = image_edit.generate_image(
                prompt,
                negative_prompt="",
                ref_images=Image.open(image_path).convert("RGB"),
                num_samples=1,
                num_steps=num_steps,
                cfg_guidance=cfg_guidance,
                seed=seed,
                show_progress=True,
                size_level=size_level,
            )[0]
    image.save(os.path.join("benchmark", image_path.split("/")[-1].replace("original" ,"edit")))
```

```bash
git clone https://huggingface.co/datasets/svjack/Genshin-Impact-Portrait-with-Tags-Filtered-IID-Gender-SP

import os
import random
from pathlib import Path
import glob

def process_folders(base_folder, num_folders=10, samples_per_folder=5):
    """
    处理文件夹并生成所需字典
    
    参数:
        base_folder: 基础文件夹路径
        num_folders: 要处理的子文件夹数量
        samples_per_folder: 每个子文件夹中抽取的三元组数量
        
    返回:
        生成的字典
    """
    # 查找所有符合条件的子文件夹
    pattern = os.path.join(base_folder, "genshin_impact_*_images_and_texts")
    matching_folders = glob.glob(pattern)
    
    # 随机选择指定数量的子文件夹
    selected_folders = random.sample(matching_folders, min(num_folders, len(matching_folders)))
    
    result_dict = {}
    
    for folder in selected_folders:
        # 获取文件夹中所有png和txt文件
        png_files = list(Path(folder).glob("*.png"))
        txt_files = list(Path(folder).glob("*.txt"))
        
        # 创建文件名到路径的映射（不带扩展名）
        txt_map = {f.stem: f for f in txt_files}
        
        # 确保有足够的文件可以抽样
        if len(png_files) < 2 or len(txt_map) < 1:
            continue
            
        # 抽取指定数量的样本
        for _ in range(samples_per_folder):
            # 确保有足够的文件可以抽样
            if len(png_files) < 2 or len(txt_map) < 1:
                break
                
            # 随机选择两个不同的图片
            img1, img2 = random.sample(png_files, 2)
            
            # 检查是否有对应的txt文件
            if img2.stem in txt_map:
                # 读取txt文件内容
                with open(txt_map[img2.stem], 'r', encoding='utf-8') as f:
                    caption = f.read().strip()
                
                # 添加到结果字典
                result_dict[str(img2.absolute())] = {
                    'ref_image_path': str(img1.absolute()),
                    'caption': caption
                }
                
                # 移除已选的文件，避免重复抽样
                png_files.remove(img1)
                png_files.remove(img2)
                del txt_map[img2.stem]
    
    return result_dict

# 使用示例
if __name__ == "__main__":
    base_folder = "Genshin-Impact-Portrait-with-Tags-Filtered-IID-Gender-SP"
    num_folders = 32  # 要处理的子文件夹数量
    samples_per_folder = 32  # 每个子文件夹中抽取的三元组数量
    
    result = process_folders(base_folder, num_folders, samples_per_folder)
    
    # 打印结果（前几个示例）
    for i, (k, v) in enumerate(result.items()):
        if i < 3:  # 只打印前3个作为示例
            print(f"{k}: {v}")
    
    print(f"\n总共处理了 {len(result)} 个三元组")

import json 
with open("change_meta.json", "w") as f:
    json.dump(result, f)

accelerate launch  --mixed_precision bf16 --num_cpu_threads_per_process 1 --num_processes 1 \
--config_file ./library/accelerate_config.yaml \
finetuning.py \
--pretrained_model_name_or_path step1x-edit-i1258.safetensors \
--qwen2p5vl Qwen2.5-VL-7B-Instruct \
--ae vae.safetensors \
--cache_latents_to_disk --save_model_as safetensors --sdpa --persistent_data_loader_workers \
--max_data_loader_n_workers 2 --seed 42 --gradient_checkpointing --mixed_precision bf16 --save_precision bf16 \
--network_module library.lora_module --network_dim 64 --network_alpha 32 --network_train_unet_only \
--optimizer_type adamw8bit --learning_rate 1e-4 \
--cache_text_encoder_outputs --cache_text_encoder_outputs_to_disk \
--highvram --max_train_epochs 100 --save_every_n_steps 500 --dataset_config step1x_edit.toml \
--output_dir change_output \
--output_name step1x-edit_change \
--timestep_sampling shift --discrete_flow_shift 3.1582 --model_prediction_type raw --guidance_scale 1.0 --fp8_base

```
