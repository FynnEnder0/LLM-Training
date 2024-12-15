from modelConfig import base_model_bnb_4b

output_dir = "./results"
per_device_train_batch_size = 2
gradient_accumulation_steps = 4
optim = "paged_adamw_32bit"
save_steps = 50
logging_steps = 5
learning_rate = 2e-4
max_grad_norm = 0.3
max_steps = 200
warmup_ratio = 0.03
lr_scheduler_type = "linear"


sft_config = SFTConfig(
    dataset_text_field="text",
    max_seq_length=512,
    output_dir=output_dir,
    per_device_train_batch_size=per_device_train_batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    optim=optim,
    save_steps=save_steps,
    logging_steps=logging_steps,
    learning_rate=learning_rate,
    fp16=True,
    max_grad_norm=max_grad_norm,
    max_steps=max_steps,
    warmup_ratio=warmup_ratio,
    group_by_length=True,
    lr_scheduler_type=lr_scheduler_type,
    gradient_checkpointing=True,
)


from trl import SFTTrainer, SFTConfig

trainer = SFTTrainer(
    model=base_model_bnb_4b,
    train_dataset=dataset,
    peft_config=peft_config,
    args=sft_config,
)