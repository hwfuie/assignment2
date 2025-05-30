import torch
import numpy as np
from datasets import load_dataset, DatasetDict, load_from_disk
from transformers import AutoTokenizer
from trl import PPOConfig, PPOTrainer, create_reference_model, AutoModelForCausalLMWithValueHead
from collections import Counter
import os
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--pretrained_model_path", type=str, default="gpt2")
parser.add_argument("--dataset_path", type=str, default=None)

args = parser.parse_args()


pre_trained_model_path = args.pretrained_model_path
dataset_path = args.dataset_path


SAVE_DIR = "pubmedqa_labeled_prompts"
def process():

    dataset = load_dataset("qiaojin/PubMedQA", "pqa_labeled", split="train")

    splits = dataset.train_test_split(test_size=0.2, seed=1234)
    splits = DatasetDict({
        "train": splits["train"],
        "test":  splits["test"]
    })

    def make_prompt(example):
        q = "Question: " + example["question"].strip() + "\n"
        contexts = example["context"]["contexts"]
        context = ""
        for c in contexts:
            context += c.strip() + " "
        q += "Context: "  + context.strip()
        return {
            "prompt"      : q,
            "long_answer": example["long_answer"].strip(),
        }

    prompt_splits = splits.map(
        make_prompt,
        remove_columns=splits["train"].column_names
    )

    os.makedirs(SAVE_DIR, exist_ok=True)
    prompt_splits.save_to_disk(SAVE_DIR)

    return prompt_splits

def load_or_process():

    if os.path.isdir(SAVE_DIR):
        ds = load_from_disk(SAVE_DIR)
    else:
        ds = process()
    return ds


def PPO(pre_trained_model,dataset_path):


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
            recall    = tp / len(r_tokens)
            f1 = 2 * precision * recall / (precision + recall)
            rewards.append(f1)
        
        return rewards


    if dataset_path is None:
        dataset = load_or_process()
    else:
        dataset = load_dataset(dataset_path)

    train_dataset = dataset["train"]


    # 1. 配置
    config = PPOConfig(
        model_name=pre_trained_model,
        learning_rate=5e-6,
        
        batch_size=2,
        mini_batch_size=1,
        ppo_epochs=4,
        
        init_kl_coef=1.0,
        target_kl=0.01,
        
        cliprange=0.2,
        cliprange_value=0.2,
    )


    model = AutoModelForCausalLMWithValueHead.from_pretrained(config.model_name)
    ref_model = create_reference_model(model)

    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    tokenizer.pad_token = tokenizer.eos_token

    ppo_trainer = PPOTrainer(
        config=config,
        model=model,
        ref_model=ref_model,
        tokenizer=tokenizer,
    )


    gen_kwargs = {
        "max_new_tokens": 32,
        "do_sample": True,
        "temperature": 0.7,
        "top_k": 0,
        "top_p": 1.0,
        "pad_token_id": tokenizer.eos_token_id
    }


    for epoch in range(config.ppo_epochs):
        idxs = np.random.permutation(len(train_dataset))
        for start in range(0, len(train_dataset), config.batch_size):
            
            batch = train_dataset.select(range(start, min(start+config.batch_size, len(train_dataset))))
            
            prompts = [ex["prompt"] for ex in batch]
            refs = [ex["long_answer"] for ex in batch]

            # Tokenize
            enc = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
            input_ids = enc.input_ids.to(ppo_trainer.accelerator.device)

            prompt_tensors = [input_ids[i, :enc.attention_mask[i].sum()] for i in range(len(input_ids))]
            outputs = ppo_trainer.generate(prompt_tensors, **gen_kwargs)

            generated = []
            response_tensors = []
            for i, out_ids in enumerate(outputs):
                prompt_len = prompt_tensors[i].size(0)
                gen_ids = out_ids[prompt_len:]
                response_tensors.append(gen_ids)
                txt = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
                generated.append(txt)

            raw_rewards = get_rewards(generated, refs)
            rewards = [torch.tensor(r).to(ppo_trainer.accelerator.device) for r in raw_rewards]

            stats = ppo_trainer.step(prompt_tensors, response_tensors, rewards)
            ppo_trainer.log_stats(stats, dict(prompt=prompts, response=generated, reference=refs), rewards)

        print(f"Epoch {epoch+1}/{config.ppo_epochs} done")

    # 8. 保存模型
    ppo_trainer.save_pretrained("ppo-gpt2-pubmedqa")


if __name__ == "__main__":
    PPO(pre_trained_model_path, dataset_path)


