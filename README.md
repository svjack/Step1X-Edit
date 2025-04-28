```bash
conda activate system

sudo apt-get update && sudo apt-get install cbm git-lfs ffmpeg

git clone https://github.com/stepfun-ai/Step1X-Edit && cd Step1X-Edit
pip install -r requirements.txt
pip install spaces
pip install -U gradio
pip install "httpx[socks]"
pip install gradio_client datasets
```

```txt
#torch>=2.3.1
torch==2.5.1
torchvision
https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.5cxx11abiFALSE-cp311-cp311-linux_x86_64.whl
liger_kernel==0.5.4
einops==0.8.1
transformers==4.49.0
qwen_vl_utils==0.0.10
safetensors==0.4.5
pillow==11.1.0
```

```bash
wget https://huggingface.co/meimeilook/Step1X-Edit-FP8/resolve/main/step1x-edit-i1258-FP8.safetensors
wget https://huggingface.co/stepfun-ai/Step1X-Edit/resolve/main/vae.safetensors

python inference.py --input_dir ./examples \
    --model_path . \
    --json_path ./examples/prompt_cn.json \
    --output_dir ./output_cn \
    --seed 1234 --size_level 1024 --offload --quantized

python gradio_demo.py

from gradio_client import Client, handle_file
client = Client("http://localhost:7860")
result = client.predict(
		prompt="Remove the person from the image.",
		ref_images=handle_file('023614aa-5b08-4fbe-8f1a-70e7e5642808 (2).png'),
		seed=-1,
		size_level=1024,
		quantized=True,
		offload=True,
		api_name="/inference"
)
print(result)

from shutil import copy2
copy2(result[0][1], result[0][1].split("/")[-1])
```

# Xiang to Luffy Demo
```python
import os
path = "Xiang_InfiniteYou_Handsome_Pics_Captioned"
os.makedirs(path, exist_ok= True)
from datasets import load_dataset
from uuid import uuid1
ds = load_dataset("svjack/Xiang_InfiniteYou_Handsome_Pics_Captioned")["train"]
for item in ds:
    uuid_name = str(uuid1())
    item["image"].save(os.path.join(path, "{}.png".format(uuid_name)))
    with open(os.path.join(path, "{}.txt".format(uuid_name)), "w") as f:
        f.write(item["joy-caption"])

import os
from tqdm import tqdm
from PIL import Image
from shutil import copy2
from gradio_client import Client, handle_file

# Set up directories
input_dir = "Xiang_InfiniteYou_Handsome_Pics_Captioned"
output_dir = "Xiang_InfiniteYou_Pose_to_Luffy_Step1X_Edit_Captioned"
os.makedirs(output_dir, exist_ok=True)

# Initialize client
client = Client("http://localhost:7860")

# Get all .png files in the input directory
png_files = [f for f in os.listdir(input_dir) if f.endswith('.png')]

# Process each file with tqdm progress bar
for png_file in tqdm(png_files, desc="Processing images"):
    # Get the base filename without extension
    base_name = os.path.splitext(png_file)[0]
    txt_file = f"{base_name}.txt"

    # Process the image through the API
    try:
        result = client.predict(
            prompt="Transform it into One Piece luffy style",
            ref_images=handle_file(os.path.join(input_dir, png_file)),
            seed=-1,
            size_level=1024,
            quantized=True,
            offload=True,
            api_name="/inference"
        )

        # Open and save the result image
        if result and len(result) > 0 and len(result[0]) > 1:
            img = Image.open(result[0][1])
            output_img_path = os.path.join(output_dir, png_file)
            img.save(output_img_path)

            # Copy the corresponding text file
            input_txt_path = os.path.join(input_dir, txt_file)
            output_txt_path = os.path.join(output_dir, txt_file)
            if os.path.exists(input_txt_path):
                copy2(input_txt_path, output_txt_path)

    except Exception as e:
        print(f"Error processing {png_file}: {str(e)}")
        continue

print("Processing complete!")
```

<div align="center">
  <img src="assets/logo.png"  height=100>
</div>
<div align="center">
  <a href="https://step1x-edit.github.io/"><img src="https://img.shields.io/static/v1?label=Project%20Page&message=Web&color=green"></a> &ensp;
  <a href="https://arxiv.org/abs/2504.17761"><img src="https://img.shields.io/static/v1?label=Tech%20Report&message=Arxiv&color=red"></a> &ensp;
  <a href="https://www.modelscope.cn/models/stepfun-ai/Step1X-Edit"><img src="https://img.shields.io/static/v1?label=Model&message=ModelScope&color=blue"></a> &ensp;
  <a href="https://discord.gg/j3qzuAyn"><img src="https://img.shields.io/static/v1?label=Discord%20Channel&message=Discord&color=purple"></a> &ensp;
  
  <a href="https://huggingface.co/stepfun-ai/Step1X-Edit"><img src="https://img.shields.io/static/v1?label=Model&message=HuggingFace&color=yellow"></a> &ensp;
  <a href="https://huggingface.co/spaces/stepfun-ai/Step1X-Edit"><img src="https://img.shields.io/static/v1?label=Online%20Demo&message=HuggingFace&color=yellow"></a> &ensp;
  <a href="https://huggingface.co/datasets/stepfun-ai/GEdit-Bench"><img src="https://img.shields.io/static/v1?label=GEdit-Bench&message=HuggingFace&color=yellow"></a> &ensp;
</div>


## 🔥🔥🔥 News!!
* Apr 27, 2025: 🎉 With community support, we update the inference code and model weights of Step1X-Edit-FP8. [meimeilook/Step1X-Edit-FP8](https://huggingface.co/meimeilook/Step1X-Edit-FP8) & [rkfg/Step1X-Edit-FP8](https://huggingface.co/rkfg/Step1X-Edit-FP8).
* Apr 26, 2025: 🎉 Step1X-Edit is now live — you can try editing images directly in the online demo! [Online Demo](https://huggingface.co/spaces/stepfun-ai/Step1X-Edit)
* Apr 25, 2025: 👋 We release the evaluation code and benchmark data of Step1X-Edit. [Download GEdit-Bench](https://huggingface.co/datasets/stepfun-ai/GEdit-Bench)
* Apr 25, 2025: 👋 We release the inference code and model weights of Step1X-Edit. [ModelScope](https://www.modelscope.cn/models/stepfun-ai/Step1X-Edit) & [HuggingFace](https://huggingface.co/stepfun-ai/Step1X-Edit) models.
* Apr 25, 2025: 👋 We have made our technical report available as open source. [Read](https://arxiv.org/abs/2504.17761)

<!-- ## Image Edit Demos -->

<div align="center">
<img width="720" alt="demo" src="assets/image_edit_demo.gif">
<p><b>Step1X-Edit:</b> a unified image editing model performs impressively on various genuine user instructions. </p>
</div>


## 🧩 Community Contributions

If you develop/use Step1X-Edit in your projects, welcome to let us know.

- FP8 model weights: [meimeilook/Step1X-Edit-FP8](https://huggingface.co/meimeilook/Step1X-Edit-FP8) by [meimeilook](https://huggingface.co/meimeilook);  [rkfg/Step1X-Edit-FP8](https://huggingface.co/rkfg/Step1X-Edit-FP8) by [rkfg](https://huggingface.co/rkfg)


## 📑 Open-source Plan
- [x] Inference & Checkpoints
- [x] Online demo (Gradio)
- [ ] Diffusers 
- [ ] Multi-gpus Sequence Parallel inference
- [x] FP8 Quantified weight
- [ ] ComfyUI



## 1. Introduction
we introduce a state-of-the-art image editing model, **Step1X-Edit**, which aims to provide comparable performance against the closed-source models like GPT-4o and Gemini2 Flash. 
More specifically, we adopt the Multimodal LLM to process the reference image and user's editing instruction. A latent embedding has been extracted and integrated with a diffusion image decoder to obtain  the target image. To train the model, we build a data generation pipeline to produce a high-quality dataset. 
For evaluation, we develop the GEdit-Bench, a novel benchmark rooted in real-world user instructions. Experimental results on GEdit-Bench demonstrate that Step1X-Edit outperforms existing open-source baselines by a substantial margin and approaches the performance of leading proprietary models, thereby making significant contributions to the field of image editing. 
More details please refer to our [technical report](https://arxiv.org/abs/2504.17761).


## 2. Model Usage
### 2.1  Requirements

The following table shows the requirements for running Step1X-Edit model (batch size = 1, w/o cfg distillation) to edit images:

|     Model    |     Peak GPU Memory (512 / 786 / 1024)  | 28 steps w flash-attn(512 / 786 / 1024) |
|:------------:|:------------:|:------------:|
| Step1X-Edit   |                42.5GB / 46.5GB / 49.8GB  | 5s / 11s / 22s |
| Step1X-Edit-FP8   |             31GB / 31.5GB / 34GB     | 6.8s / 13.5s / 25s | 
| Step1X-Edit + offload   |       25.9GB / 27.3GB / 29.1GB | 49.6s / 54.1s / 63.2s |
| Step1X-Edit-FP8 + offload   |   18GB / 18GB / 18GB | 35s / 40s / 51s |

* The model is tested on one H800 GPUs.
* We recommend to use GPUs with 80GB of memory for better generation quality and efficiency.
* The Step1X-Edit-FP8 model we tested comes from [meimeilook/Step1X-Edit-FP8](https://huggingface.co/meimeilook/Step1X-Edit-FP8).


### 2.2 Dependencies and Installation


python >=3.10.0 and install [torch](https://pytorch.org/get-started/locally/) >= 2.2 with cuda toolkit and corresponding torchvision. We test our model using torch==2.3.1 and torch==2.5.1 with cuda-12.1.


Install requirements:
  
``` bash
pip install -r requirements.txt
```

Install [`flash-attn`](https://github.com/Dao-AILab/flash-attention), here we provide a script to help find the pre-built wheel suitable for your system. 
    
```bash
python scripts/get_flash_attn.py
```

The script will generate a wheel name like `flash_attn-2.7.2.post1+cu12torch2.5cxx11abiFALSE-cp310-cp310-linux_x86_64.whl`, which could be found in [the release page of flash-attn](https://github.com/Dao-AILab/flash-attention/releases).

Then you can download the corresponding pre-built wheel and install it following the instructions in [`flash-attn`](https://github.com/Dao-AILab/flash-attention).


### 2.3 Inference Scripts
After downloading the [model weights](https://huggingface.co/stepfun-ai/Step1X-Edit), you can use the following scripts to edit images:

```
bash scripts/run_examples.sh
```
The default script runs the inference code with non-quantified weights. If you want to save the GPU memory usage, you can 1) download the FP8 weights and set the `--quantized` flag in the script, or 2) set the `--offload` flag in the script to offload some modules to CPU.

This default script runs the inference code on example inputs. The results will look like:
<div align="center">
<img width="1080" alt="results" src="assets/results_show.png">
</div>

## 3. Benchmark
We release [GEdit-Bench](https://huggingface.co/datasets/stepfun-ai/GEdit-Bench) as a new benchmark, grounded in real-world usages is developed to support more authentic and comprehensive evaluation. This benchmark, which is carefully curated to reflect actual user editing needs and a wide range of editing scenarios, enables more authentic and comprehensive evaluations of image editing models.
The evaluation process and related code can be found in [GEdit-Bench/EVAL.md](GEdit-Bench/EVAL.md). Part results of the benchmark are shown below:
<div align="center">
<img width="1080" alt="results" src="assets/eval_res_en.png">
</div>


## 4. Citation
```
@article{liu2025step1x-edit,
      title={Step1X-Edit: A Practical Framework for General Image Editing}, 
      author={Shiyu Liu and Yucheng Han and Peng Xing and Fukun Yin and Rui Wang and Wei Cheng and Jiaqi Liao and Yingming Wang and Honghao Fu and Chunrui Han and Guopeng Li and Yuang Peng and Quan Sun and Jingwei Wu and Yan Cai and Zheng Ge and Ranchen Ming and Lei Xia and Xianfang Zeng and Yibo Zhu and Binxing Jiao and Xiangyu Zhang and Gang Yu and Daxin Jiang},
      journal={arXiv preprint arXiv:2504.17761},
      year={2025}
}
```

## 5. Acknowledgement
We would like to express our sincere thanks to the contributors of [SD3](https://huggingface.co/stabilityai/stable-diffusion-3-medium), [FLUX](https://github.com/black-forest-labs/flux), [Qwen](https://github.com/QwenLM/Qwen2.5), [diffusers](https://github.com/huggingface/diffusers) and [HuggingFace](https://huggingface.co) teams, for their open research and exploration.


## 6. Disclaimer
The results produced by this image editing model are entirely determined by user input and actions. The development team and this open-source project are not responsible for any outcomes or consequences arising from its use.

## 7. LICENSE
Step1X-Edit is licensed under the Apache License 2.0. You can find the license files in the respective github and  HuggingFace repositories.
