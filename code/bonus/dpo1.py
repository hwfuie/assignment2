import os
import argparse
import random
import torch
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import DPOConfig, DPOTrainer

# 解析参数
parser = argparse.ArgumentParser()

parser.add_argument("--pretrained_model_path", type=str, default="gpt2")
parser.add_argument("--dataset_path", type=str, default=None)
parser.add_argument("--output_dir", type=str, default="dpo-gpt2-pubmedqa")  # <== 添加这行

args = parser.parse_args()

pre_trained_model_path = args.pretrained_model_path
dataset_path = args.dataset_path
output_dir = args.output_dir  # <== 使用这个值

# DPO 配对样本构造
def make_dpo_dataset(train_dataset, seed=42):
    random.seed(seed)
    all_answers = [ex["long_answer"] for ex in train_dataset]

    def sample_pair(example, idx):
        chosen = example["long_answer"]
        rejected = chosen
        while rejected == chosen:
            rejected = random.choice(all_answers)
        return {
            "prompt": example["prompt"],
            "chosen": chosen,
            "rejected": rejected
        }
    paired = train_dataset.map(sample_pair, with_indices=True)
    paired = paired.remove_columns([col for col in paired.column_names if col not in ["prompt", "chosen", "rejected"]])
    return paired

# 主训练逻辑
def train_dpo(model_name: str, dataset_path: str, output_dir: str):
    # 加载预处理数据
    dataset = load_from_disk(dataset_path)
    train_dataset = make_dpo_dataset(dataset["train"])

    # 模型与 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name, local_files_only=True)

    # 配置
    config = DPOConfig(
        output_dir=output_dir,
        per_device_train_batch_size=4,
        learning_rate=5e-6,
        num_train_epochs=3,
        logging_steps=10,
        save_strategy="epoch",
        report_to="none"
    )

    trainer = DPOTrainer(
        model=model,
        args=config,
        processing_class=tokenizer,
        train_dataset=train_dataset,
    )
    # 开始训练
    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

if __name__ == "__main__":
    train_dpo(args.pretrained_model_path, args.dataset_path, args.output_dir)
	
