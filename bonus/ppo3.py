import os
import argparse
import random
import torch
import numpy as np
from collections import Counter
from datasets import load_from_disk
from transformers import AutoTokenizer
from trl import PPOConfig, PPOTrainer, create_reference_model, AutoModelForCausalLMWithValueHead

# ===================== 参数解析 =====================
parser = argparse.ArgumentParser()
parser.add_argument("--pretrained_model_path", type=str, required=True)
parser.add_argument("--dataset_path", type=str, required=True)
parser.add_argument("--output_dir", type=str, default="ppo-gpt2-pubmedqa")
args = parser.parse_args()

# ===================== 奖励函数 =====================
def get_rewards(preds, refs):
    rewards = []
    for p, r in zip(preds, refs):
        p_tokens = p.split()
        r_tokens = r.split()
        pc = Counter(p_tokens)
        rc = Counter(r_tokens)
        tp = sum(min(pc[t], rc[t]) for t in pc)
        if tp == 0:
            rewards.append(0.0)
            continue
        precision = tp / len(p_tokens)
        recall = tp / len(r_tokens)
        f1 = 2 * precision * recall / (precision + recall)
        rewards.append(f1)
    return rewards

# ===================== PPO 主函数 =====================
def run_ppo(model_path, dataset_path, output_dir):
    dataset = load_from_disk(dataset_path)
    train_dataset = dataset["train"]
    
    config = PPOConfig(
        learning_rate=5e-6,
        
        batch_size=2,
        mini_batch_size=1,
        ppo_epochs=4,
        
        init_kl_coef=1.0,
        target_kl=0.01,
        
        cliprange=0.2,
        cliprange_value=0.2,
    )
    model = AutoModelForCausalLMWithValueHead.from_pretrained(model_path)
    ref_model = create_reference_model(model)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    ppo_trainer = PPOTrainer(
        config=config,
        model=model,
        ref_model=ref_model,
        tokenizer=tokenizer,
    )
    gen_kwargs = {
        "max_new_tokens": 100,
        "do_sample": True,
        "temperature": 0.7,
        "top_k": 0,
        "top_p": 1.0,
        "pad_token_id": tokenizer.eos_token_id
    }

    NUM_EPOCHS = 4
    for epoch in range(NUM_EPOCHS):
        idxs = np.random.permutation(len(train_dataset))
        for start in range(0, len(train_dataset), config.batch_size):
            batch = train_dataset.select(range(start, min(start + config.batch_size, len(train_dataset))))
            prompts = [ex["prompt"] for ex in batch]
            refs = [ex["long_answer"] for ex in batch]

            enc = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
            input_ids = enc.input_ids.to(ppo_trainer.accelerator.device)
            attention_mask = enc.attention_mask

            prompt_tensors = [input_ids[i, :attention_mask[i].sum().item()] for i in range(len(input_ids))]

            outputs = ppo_trainer.generate(prompt_tensors, **gen_kwargs)

            generated = []
            response_tensors = []
            for i, out_ids in enumerate(outputs):
                prompt_len = prompt_tensors[i].size(0)
                gen_ids = out_ids[prompt_len:]
                response_tensors.append(gen_ids.to(ppo_trainer.accelerator.device))
                txt = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
                generated.append(txt)

                print(f"[PROMPT] {prompts[i]}")
                print(f"[GENERATION] {txt}")
                print(f"[REFERENCE] {refs[i]}")

            raw_rewards = get_rewards(generated, refs)
            rewards = [torch.tensor(r).to(ppo_trainer.accelerator.device) for r in raw_rewards]

            stats = ppo_trainer.step(prompt_tensors, response_tensors, rewards)
            ppo_trainer.log_stats(stats, dict(prompt=prompts, response=generated, reference=refs), rewards)

        print(f"Epoch {epoch+1}/{NUM_EPOCHS} done")

    ppo_trainer.model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

if __name__ == "__main__":
    run_ppo(args.pretrained_model_path, args.dataset_path, args.output_dir)
