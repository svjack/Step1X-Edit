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
