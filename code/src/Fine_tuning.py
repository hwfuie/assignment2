# Fine tuning
import os
os.environ["USE_TF"] = "0"
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import load_dataset
import torch

from transformers import GPT2Tokenizer, GPT2LMHeadModel

print(torch.cuda.is_available())

model_name = "gpt2"
data_file = "/home/stu4/assignment2/code/data/pubmedqa_all.jsonl"
output_dir = "/home/stu4/assignment2/code/data"

tokenizer = GPT2Tokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
model = GPT2LMHeadModel.from_pretrained(model_name)
model.config.pad_token_id = tokenizer.pad_token_id

dataset = load_dataset("json", data_files={"train": data_file})
full_dataset=dataset["train"]

def preprocess(example):
    full = example["prompt"] + " " + example["response"]
    tokenized = tokenizer(full, truncation=True, padding="max_length", max_length=512)
    tokenized["labels"] = tokenized["input_ids"].copy()  # 将 input_ids 作为 labels 用于训练
    return tokenized

tokenized_dataset = full_dataset.map(
    preprocess,
    remove_columns=full_dataset.column_names  # 删除原始字段，仅保留 input_ids、attention_mask、labels
)

training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=8,
    num_train_epochs=2,
    learning_rate=5e-5,
    save_steps=500,
    logging_steps=20,
    fp16=torch.cuda.is_available(),
    save_total_limit=1,
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
    data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
)

trainer.train()

trainer.save_model(output_dir)
tokenizer.save_pretrained(output_dir)
print("微调完成，模型保存在:", output_dir)
