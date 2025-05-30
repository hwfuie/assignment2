import os
import argparse
import random
import torch
from datasets import load_dataset, load_from_disk, DatasetDict
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import DPOConfig, DPOTrainer

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


def make_dpo_dataset(train_dataset, seed = 42):

    random.seed(seed)
    def sample_pair(example, idx, all_answers):

        chosen = example["long_answer"]

        rejected = chosen
        while rejected == chosen:
            rejected = random.choice(all_answers)
        return {"chosen": chosen, "rejected": rejected}

    all_answers = [ex["long_answer"] for ex in train_dataset]
    
    paired = train_dataset.map(
        lambda ex, idx: sample_pair(ex, idx, all_answers),
        with_indices=True
    )

    paired = paired.remove_columns([c for c in paired.column_names if c not in ["prompt","chosen","rejected"]])

    return paired


def train_dpo(model_name: str, dataset_path: str = None):
    # 1) 加载或预处理数据
    ds = load_or_process() if dataset_path is None else load_from_disk(dataset_path)
    ds_train = make_dpo_dataset(ds['train'])

    # 2) Tokenizer & Model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name)

    config = DPOConfig(output_dir="dpo-gpt2-pubmedqa", logging_steps=10)

    trainer = DPOTrainer(
        model,
        args=config,
        processing_class=tokenizer,
        train_dataset=ds_train,
    )

    trainer.train()
    trainer.save_model()
    

if __name__ == "__main__":
    train_dpo(pre_trained_model_path, dataset_path)