def run_ppo(model_path, dataset_path, output_dir):
    dataset = load_from_disk(dataset_path)
    train_dataset = dataset["train"]
    
    config = PPOConfig(
        learning_rate=5e-6,
        batch_size=2,
        mini_batch_size=1,
        num_ppo_epochs=4,
        kl_coef=0.1,
        cliprange=0.2,
        vf_coef=0.1,
        gamma=1.0,
        lam=0.95,
    )

    model = AutoModelForCausalLMWithValueHead.from_pretrained(model_path)
    ref_model = create_reference_model(model)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    ppo_trainer = PPOTrainer(
        model=model,
        ref_model=ref_model,
        reward_model=model,
        train_dataset=train_dataset,
        **vars(config)
    )

    gen_kwargs = {
        "max_new_tokens": 32,
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
