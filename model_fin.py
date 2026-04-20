import os
import json
import torch
import shutil
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.modifiers.smoothquant import SmoothQuantModifier 

MODEL_ID = "./base_model"     
OUT_DIR  = "./model"          
EVAL_OUT_DIR = "./eval_results"

# 1. 양자화 설정 
DATASET_ID = "LGAI-EXAONE/MANTA-1M"
DATASET_SPLIT = "train"
NUM_CALIBRATION_SAMPLES = 256
MAX_SEQUENCE_LENGTH = 512

TARGETS = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj"     
]
IGNORE  = ["embed_tokens", "lm_head"]

# 2. 평가 데이터셋 설정 
EVAL_TASKS = {
    "QA_Knowledge": {
        "id": "nlpai-lab/kullm-v2",
        "split": "train",
        "samples": 5,
        "prompt_col": "instruction"
    },
    "Instruction_Following": {
        "id": "google/IFEval",
        "split": "train",
        "samples": 5,
        "prompt_col": "prompt"
    }
}

print("[INFO] 모델 로드 중...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_ID,
    trust_remote_code=True,
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

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

print("[INFO] W8A8 및 KV Cache FP8 최적화 양자화 시작...")

recipe = [
    SmoothQuantModifier(
        smoothing_strength=0.5, 
        mappings=[
            (["q_proj", "k_proj", "v_proj"], "input_layernorm"),
            (["o_proj"], "post_attention_layernorm"),
            (["gate_proj", "up_proj"], "post_attention_layernorm"),
            (["down_proj"], "post_mlp_layernorm")
        ] 
    ),
    QuantizationModifier(
        targets=TARGETS,
        ignore=IGNORE,
        scheme="W8A8", 
    )
]

oneshot(
    model=model,
    dataset=ds,
    recipe=recipe,
    max_seq_length=MAX_SEQUENCE_LENGTH,
    num_calibration_samples=NUM_CALIBRATION_SAMPLES,
)

print("[INFO] 양자화 연산 완료")

# KV Cache FP8 메타데이터 주입
if hasattr(model.config, "quantization_config"):
    model.config.quantization_config["kv_cache_scheme"] = "fp8"
else:
    model.config.kv_cache_scheme = "fp8"

os.makedirs(OUT_DIR, exist_ok=True)
model.save_pretrained(OUT_DIR, save_compressed=True)
tokenizer.save_pretrained(OUT_DIR)
print(f"[INFO] 모델 저장 완료: {OUT_DIR}")

# 3. 양자화 모델 평가
print("[INFO] 양자화된 모델에 대한 Evaluation 추론 테스트 시작...")
os.makedirs(EVAL_OUT_DIR, exist_ok=True)

model.eval()

for task_name, config in EVAL_TASKS.items():
    print(f"\n[EVAL] 데이터셋 로드 중: {config['id']} ({task_name})")
    eval_ds = load_dataset(config['id'], split=config['split'])
    
    results = []
    
    for i in range(min(config['samples'], len(eval_ds))):
        sample = eval_ds[i]
        user_prompt = sample[config['prompt_col']]
        
        # 모델 포맷에 맞게 챗 템플릿 적용
        messages = [{"role": "user", "content": user_prompt}]
        input_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        
        inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs, 
                max_new_tokens=256, 
                temperature=0.1, 
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
            
        generated_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        
        results.append({
            "task": task_name,
            "prompt": user_prompt,
            "generated": generated_text.strip()
        })
        
    # 평가 결과 저장
    output_path = os.path.join(EVAL_OUT_DIR, f"{task_name}_results.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
        
    print(f"[EVAL] {task_name} 테스트 완료. 결과가 {output_path}에 저장되었습니다.")

# 4. 제출용 패키징
zip_name = "baseline_submit"
print(f"\n[INFO] {zip_name}.zip 생성 중 (모델 및 평가 결과 포함)...")

# 평가 결과도 함께 확인할 수 있도록 zip에 포함
if not os.path.exists(os.path.join(OUT_DIR, "eval_results")):
    shutil.copytree(EVAL_OUT_DIR, os.path.join(OUT_DIR, "eval_results"))

shutil.make_archive(
    base_name=zip_name,
    format="zip",
    root_dir=".",
    base_dir=OUT_DIR,
)

print(f"[INFO] 생성 완료: {zip_name}.zip")