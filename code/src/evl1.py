import torch
import numpy as np
import time
import evaluate
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForCausalLM


# ========== 路径配置 ==========

original_model_path = "/home/stu4/assignment2/code/data/checkpoint-52818"

# 微调后模型路径
trained_model_path = "/home/stu4/assignment2/dpo_gpt2_pubmedqa"

# 数据集路径（load_from_disk 格式）
test_dataset_path = "/home/stu4/assignment2/code/data/pubmedqa_labeled_prompts"


# ========== 模型加载 ==========
def load_model_and_tokenizer(path):
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    
    model = AutoModelForCausalLM.from_pretrained(path, local_files_only=True)
    return model.to(device), tokenizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
original_model, original_tokenizer = load_model_and_tokenizer(original_model_path)
trained_model, trained_tokenizer = load_model_and_tokenizer(trained_model_path)


# ========== 数据加载 ==========
dataset = load_from_disk(test_dataset_path)
test_dataset = dataset["test"]


# ========== 生成函数 ==========
def generate_answer(model, tokenizer, prompt, max_new_tokens=32, temperature=0.7):
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    return tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()


# ========== 定量评估 ==========
bleu = evaluate.load("bleu")
rouge = evaluate.load("rouge")

def evaluate_model(model, tokenizer, dataset, max_samples=100):
    generated, references, prompts = [], [], []

    indices = np.random.choice(len(dataset), max_samples, replace=False)
    for i in indices:
        sample = dataset[int(i)]
        prompt = sample["prompt"]
        reference = sample["long_answer"]
        answer = generate_answer(model, tokenizer, prompt)
        
        prompts.append(prompt)
        references.append([reference])
        generated.append(answer)

    bleu_result = bleu.compute(predictions=generated, references=references)
    rouge_result = rouge.compute(predictions=generated, references=[r[0] for r in references])

    return {
        "bleu": bleu_result,
        "rouge": rouge_result,
        "prompts": prompts,
        "references": references,
        "generated": generated
    }


# ========== 样例输出 ==========
def qualitative_comparison(model1, tokenizer1, model2, tokenizer2, dataset, num_samples=5):
    indices = np.random.choice(len(dataset), num_samples, replace=False)
    for i in indices:
        sample = dataset[int(i)]
        prompt = sample["prompt"]
        ref = sample["long_answer"]
        gen1 = generate_answer(model1, tokenizer1, prompt)
        gen2 = generate_answer(model2, tokenizer2, prompt)

        print("="*80)
        print(f"[Prompt] {prompt}\n[Reference] {ref}")
        print(f"[Original model] {gen1}")
        print(f"[Trained  model] {gen2}")
        print("="*80)


# ========== 推理速度和回答长度 ==========
def evaluate_speed_length(model, tokenizer, dataset, max_samples=50):
    lengths, total_time = [], 0.0
    indices = np.random.choice(len(dataset), max_samples, replace=False)
    for i in indices:
        prompt = dataset[int(i)]["prompt"]
        start = time.time()
        answer = generate_answer(model, tokenizer, prompt)
        total_time += time.time() - start
        lengths.append(len(answer.split()))
    return sum(lengths)/len(lengths), total_time/len(lengths)


# ========== 主评估流程 ==========
print(">>> Evaluating Original Model")
res1 = evaluate_model(original_model, original_tokenizer, test_dataset)
print(f"BLEU: {res1['bleu']['bleu']:.4f}")
print(f"ROUGE-L: {res1['rouge']['rougeL']:.4f}")

print(">>> Evaluating Trained Model")
res2 = evaluate_model(trained_model, trained_tokenizer, test_dataset)
print(f"BLEU: {res2['bleu']['bleu']:.4f}")
print(f"ROUGE-L: {res2['rouge']['rougeL']:.4f}")

print("\n>>> Sample Comparison")
qualitative_comparison(original_model, original_tokenizer, trained_model, trained_tokenizer, test_dataset, num_samples=5)

print("\n>>> Speed and Length")
l1, t1 = evaluate_speed_length(original_model, original_tokenizer, test_dataset)
l2, t2 = evaluate_speed_length(trained_model, trained_tokenizer, test_dataset)
print(f"Original - Avg Length: {l1:.2f}, Time: {t1:.4f}s")
print(f"Trained  - Avg Length: {l2:.2f}, Time: {t2:.4f}s")
