import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    pipeline,
    logging,
)
from peft import LoraConfig
from trl import SFTConfig
from trl import SFTTrainer



base_model = "Meta-Llama-3.1-8B-Instruct" # the load folder

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(base_model)

# BNB configuration
base_model_bnb_4b = AutoModelForCausalLM.from_pretrained(
    base_model,
    quantization_config=bnb_config,
    device_map='auto'
)

lora_r = 16
lora_alpha = 16
lora_dropout = 0.01 #oder 0, da wenig trainingsdaten

peft_config = LoraConfig(
    lora_alpha=lora_alpha,
    lora_dropout=lora_dropout,
    r=lora_r,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
)
#Lora Adaptor



#SFT_Trainer

dataset = load_dataset(r"/Users/ender/PycharmProjects/PythonProject1/Files/train_data_full_final.jsonl")

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

trainer = SFTTrainer(
    model=base_model_bnb_4b,
    train_dataset=dataset,
    peft_config=peft_config,
    args=sft_config,
)


trainer.train()