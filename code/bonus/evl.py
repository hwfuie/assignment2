import torch
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import AutoModelForCausalLMWithValueHead
import evaluate
import time
from datasets import load_from_disk


# 1. 加载原始模型和训练好的模型
original_model_name = "/home/stu4/assignment2/dpo_gpt2_pubmedqa"
original_tokenizer = AutoTokenizer.from_pretrained(original_model_name,local_files_only=True)
original_tokenizer.pad_token = original_tokenizer.eos_token
original_model = AutoModelForCausalLM.from_pretrained(original_model_name)

trained_model_path =  "/home/stu4/assignment2/code/data/checkpoint-52818"
trained_tokenizer = AutoTokenizer.from_pretrained(trained_model_path,local_files_only=True)
trained_tokenizer.pad_token = trained_tokenizer.eos_token
trained_model = AutoModelForCausalLM.from_pretrained(trained_model_path)

# 确保模型在相同的设备上运行
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
original_model.to(device)
trained_model.to(device)

# 2. 准备测试数据
test_dataset_path = "/home/stu4/assignment2/code/data/pubmedqa_labeled_prompts"
test_dataset = load_from_disk(test_dataset_path)["test"]


# 3. 定义生成函数
def generate_answer(model, tokenizer, prompt, max_new_tokens=32, temperature=0.7):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, padding=True)
    input_ids = inputs.input_ids.to(device)

    outputs = model.generate(
        input_ids,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )

    answer = tokenizer.decode(outputs[0, input_ids.shape[1]:], skip_special_tokens=True).strip()
    return answer




# 4. 定义评估函数
bleu = evaluate.load("bleu")
rouge = evaluate.load("rouge")

def evaluate_model(model, tokenizer, dataset, max_samples=100):
    generated_answers = []
    reference_answers = []
    prompts = []

    indices = np.random.choice(len(dataset), max_samples, replace=True)
    for idx in indices:
        sample = dataset[int(idx)]
        prompt = sample["prompt"]
        reference_answer = sample["long_answer"]

        generated_answer = generate_answer(model, tokenizer, prompt, max_new_tokens=32, temperature=0.7)

        generated_answers.append(generated_answer)
        reference_answers.append([reference_answer])
        prompts.append(prompt)

    bleu_result = bleu.compute(predictions=generated_answers, references=reference_answers)
    rouge_result = rouge.compute(predictions=generated_answers, references=reference_answers)

    return {
        "bleu": bleu_result,
        "rouge": rouge_result,
        "generated_answers": generated_answers,
        "reference_answers": reference_answers,
        "prompts": prompts
    }

# 5. 评估原始模型和训练好的模型
original_results = evaluate_model(original_model, original_tokenizer, test_dataset)
print("Original Model Results:")
print(f"BLEU: {original_results['bleu']}")
print(f"ROUGE: {original_results['rouge']}")

trained_results = evaluate_model(trained_model, trained_tokenizer, test_dataset)
print("Trained Model Results:")
print(f"BLEU: {trained_results['bleu']}")
print(f"ROUGE: {trained_results['rouge']}")

# 6. 定性评估并输出样例
def qualitative_evaluation(model, tokenizer, dataset, num_samples=5):
    indices = np.random.choice(len(dataset), num_samples, replace=False)
    samples = []
    for idx in indices:
        sample = dataset[int(idx)]
        prompt = sample["prompt"]
        reference_answer = sample["long_answer"]

        generated_answer = generate_answer(model, tokenizer, prompt)

        samples.append({
            "prompt": prompt,
            "reference_answer": reference_answer,
            "generated_answer": generated_answer
        })

    return samples

# 获取原始模型和训练模型的样例输出
original_samples = qualitative_evaluation(original_model, original_tokenizer, test_dataset, num_samples=5)
trained_samples = qualitative_evaluation(trained_model, trained_tokenizer, test_dataset, num_samples=5)

# 打印样例对比
print("\n对比原始模型和训练模型的样例输出：")
for i in range(5):
    print(f"样例 {i+1}:")
    print(f"Prompt:\n{original_samples[i]['prompt']}")
    print(f"参考答案: {original_samples[i]['reference_answer']}")
    print(f"原始模型生成: {original_samples[i]['generated_answer']}")
    print(f"训练模型生成: {trained_samples[i]['generated_answer']}")
    print('-' * 80)

# 7. 评估回答的长度和推理速度
def evaluate_answer_length_and_speed(model, tokenizer, dataset, max_samples=100):
    lengths = []
    total_time = 0

    indices = np.random.choice(len(dataset), max_samples, replace=False)
    for idx in indices:
        sample = dataset[int(idx)]
        prompt = sample["prompt"]

        start_time = time.time()
        generated_answer = generate_answer(model, tokenizer, prompt)
        end_time = time.time()

        lengths.append(len(generated_answer.split()))
        total_time += (end_time - start_time)

    avg_length = sum(lengths) / len(lengths)
    avg_time = total_time / len(indices)

    return avg_length, avg_time

original_avg_length, original_avg_time = evaluate_answer_length_and_speed(original_model, original_tokenizer, test_dataset)
print(f"原始模型 - 平均回答长度: {original_avg_length:.2f}")
print(f"原始模型 - 平均推理时间: {original_avg_time:.4f} 秒")

