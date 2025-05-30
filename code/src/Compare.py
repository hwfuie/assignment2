from transformers import GPT2LMHeadModel, GPT2Tokenizer, pipeline

# 原始 GPT-2 模型
base_model = GPT2LMHeadModel.from_pretrained("gpt2")
base_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
base_tokenizer.pad_token = base_tokenizer.eos_token
base_model.config.pad_token_id = base_tokenizer.pad_token_id

# 微调后模型路径
finetuned_model_path = "/home/stu4/assignment2/code/data"

# 微调后的模型
ft_model = GPT2LMHeadModel.from_pretrained(finetuned_model_path)
ft_tokenizer = GPT2Tokenizer.from_pretrained(finetuned_model_path)
ft_tokenizer.pad_token = ft_tokenizer.eos_token
ft_model.config.pad_token_id = ft_tokenizer.pad_token_id

# 测试输入
prompt="Pubmed Question: How can you describe a table?"
# 构造 pipeline
pipe_base = pipeline("text-generation", model=base_model, tokenizer=base_tokenizer)
pipe_ft = pipeline("text-generation", model=ft_model, tokenizer=ft_tokenizer)

# 生成结果
output_base = pipe_base(prompt, max_length=100, do_sample=True,temperature=0.8,top_p=0.9,repetition_penalty=1.2)[0]["generated_text"]
output_ft = pipe_ft(prompt, max_length=100, do_sample=True,temperature=0.8,top_p=0.9,repetition_penalty=1.2)[0]["generated_text"]

# 输出比较结果
print("=== 原始模型输出 ===")
print(output_base)
print("\n=== 微调后模型输出 ===")
print(output_ft)
