import requests
import torch
from PIL import Image
import argparse
import torch

from transformers import AutoProcessor, LlavaForConditionalGeneration

# model_id = "llava-hf/llava-1.5-7b-hf"
model_id = "/root/autodl-tmp/model/MLLM/llava-v1.5-7b"

prompt_1 = "USER: <image>\nWhat does this image show?\nASSISTANT:"
prompt_2 = "USER: <image> <image> \nWhat is the difference between these two images?\nASSISTANT:"
image_file_1 = "/root/merlion.png"
image_file_2 = "/root/black_image.png"

# disable_torch_init()

# from llava.model.builder import load_pretrained_model
# from llava.mm_utils import get_model_name_from_path

# model_path = "/root/autodl-tmp/model/MLLM/llava-v1.5-7b-lora"
# model_base = "/root/autodl-tmp/model/LLM/vicuna-7b-v1.5"

# tokenizer, model, image_processor, context_len = load_pretrained_model(
    # model_path=model_path,
    # model_base=model_base,
    # model_name=get_model_name_from_path(model_path)
# )

model = LlavaForConditionalGeneration.from_pretrained(model_id, torch_dtype=torch.float16, low_cpu_mem_usage=True, use_flash_attention_2=True).to(0)
processor = AutoProcessor.from_pretrained(model_id)
raw_image_1 = Image.open(image_file_1)
raw_image_2 = Image.open(image_file_2)
inputs = processor([prompt_1, prompt_2], [raw_image_1, raw_image_1, raw_image_2], padding=True, return_tensors="pt").to(0, torch.float16)
import pdb

pdb.set_trace()
output = model.generate(**inputs, max_new_tokens=200, do_sample=False)
print(processor.batch_decode(output, skip_special_tokens=True))