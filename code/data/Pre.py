from datasets import load_dataset, DatasetDict
import os

SAVE_DIR = "/home/stu4/assignment2/data/pubmedqa_labeled_prompts"

def make_prompt(example):
    q = "Question: " + example["question"].strip() + "\n"
    context = " ".join(example["context"]["contexts"])
    return {
        "prompt": q + "Context: " + context.strip(),
        "long_answer": example["long_answer"].strip(),
    }

dataset = load_dataset("qiaojin/PubMedQA", "pqa_labeled", split="train")
splits = dataset.train_test_split(test_size=0.2, seed=1234)
prompt_splits = DatasetDict({
    "train": splits["train"].map(make_prompt, remove_columns=splits["train"].column_names),
    "test":  splits["test"].map(make_prompt, remove_columns=splits["test"].column_names),
})

os.makedirs(SAVE_DIR, exist_ok=True)
prompt_splits.save_to_disk(SAVE_DIR)
print(f"✅ 数据保存成功: {SAVE_DIR}")
