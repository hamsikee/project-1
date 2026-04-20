# %% [markdown]
# # Import
# 

# %%

import os
import torch
import shutil
from pathlib import Path

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier


# %% [markdown]
# 
# # Setting
# 

# %%

MODEL_ID = r"C:\Users\user\Desktop\LG Aimers\open\base_model"     
OUT_DIR  = "./model"          

DATASET_ID = "LGAI-EXAONE/MANTA-1M"
DATASET_SPLIT = "train"

NUM_CALIBRATION_SAMPLES = 1024
MAX_SEQUENCE_LENGTH = 1024


# %% [markdown]
# 
# # Quantization
# 

# %%

SCHEME = "W8A8"
TARGETS = ["Linear"]
IGNORE  = ["lm_head"]


# %% [markdown]
# 
# # Model Loads
# 

# %%

print("[INFO] 모델 로드 중...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_ID,
    trust_remote_code=True,
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    device_map="cuda",
    trust_remote_code=True
)

print("[INFO] 모델/토크나이저 로드 완료")


# %% [markdown]
# 
# # Dataset Loads & Preprocess
# 

# %%

print("[INFO] 캘리브레이션 데이터 로드 중...")

ds = load_dataset(
    DATASET_ID,
    split=f"{DATASET_SPLIT}[:{NUM_CALIBRATION_SAMPLES}]",
)

def preprocess(example):
    return {
        "text": tokenizer.apply_chat_template(
            example["conversations"],
            add_generation_prompt=True,
            tokenize=False)
    }

ds = ds.map(preprocess)

print("[INFO] 데이터 전처리 완료")


# %% [markdown]
# 
# # GPTQ Quantization
# 

# %%

print(f"[INFO] SmoothQuant 시작 (scheme={SCHEME}, samples={NUM_CALIBRATION_SAMPLES}, max_len={MAX_SEQUENCE_LENGTH})...")

recipe = [
    QuantizationModifier(
        scheme="W8A8",
        targets=TARGETS,
        ignore=IGNORE,
    )
]

oneshot(
    model=model,
    dataset=ds,
    recipe=recipe,
    max_seq_length=MAX_SEQUENCE_LENGTH,
    num_calibration_samples=NUM_CALIBRATION_SAMPLES,
)

print("[INFO] GPTQ 완료")


# %% [markdown]
# 
# # Model Save
# 

# %%

os.makedirs(OUT_DIR, exist_ok=True)

model.save_pretrained(OUT_DIR, save_compressed=True)
tokenizer.save_pretrained(OUT_DIR)

print(f"[INFO] 모델 저장 완료: {OUT_DIR}")


# %% [markdown]
# 
# # Submission
# 

# %%

zip_name = "submit"
print(f"[INFO] {zip_name}.zip 생성 중...")

shutil.make_archive(
    base_name=zip_name,
    format="zip",
    root_dir=".",
    base_dir=OUT_DIR,
)

print(f"[INFO] 생성 완료: {zip_name}.zip")