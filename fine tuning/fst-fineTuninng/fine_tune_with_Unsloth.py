import unsloth     
from unsloth import FastModel, UnslothTrainer, UnslothTrainingArguments
from datasets import load_dataset
from transformers import DataCollatorForSeq2Seq
from peft import LoraConfig, get_peft_model, TaskType

# 1. Load the correct 1B instruct model in 4-bit
model, tokenizer = FastModel.from_pretrained(
    "unsloth/Llama-3.2-1B-Instruct-bnb-4bit",
    load_in_4bit=True,
)

# 2. Attach LoRA so there are trainable parameters
lora_config = LoraConfig(
    task_type=TaskType.SEQ_2_SEQ_LM,
    inference_mode=False,
    r=8, # Parametre mystere
    lora_alpha=8,
    lora_dropout=0.05,
)
model = get_peft_model(model, lora_config)

# 3. Batched formatting function
def formatting_func(examples):
    return [
        f"### Question:\n{q}\n\n### Answer:\n{a}"
        for q, a in zip(examples["question"], examples["answer"])
    ]

# 4. Load your JSONL
dataset = load_dataset("json", data_files={"train": "fusion_fr_dataset.jsonl"}, split="train")

# 5. Training args
args = UnslothTrainingArguments(
    output_dir="outputs",
    per_device_train_batch_size=8,
    num_train_epochs=3,
    save_strategy="steps",
    save_steps=100,
    learning_rate=5e-5,
    embedding_learning_rate=5e-6,
)

# 6. Data collator
data_collator = DataCollatorForSeq2Seq(tokenizer)

# 7. Instantiate & run
trainer = UnslothTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    data_collator=data_collator,
    args=args,
    formatting_func=formatting_func,
    max_seq_length=2048,
    dataset_num_proc=4,
)

trainer.train()

trainer.save_model("lora-1b-instruct-adapter")
tokenizer.save_pretrained("lora-1b-instruct-adapter")
print("Adapter and tokenizer saved to ./lora-1b-instruct-adapter/")