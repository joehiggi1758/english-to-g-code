from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Trainer, TrainingArguments
from datasets import Dataset
import pandas as pd

# Load dataset
data = {
    "instruction": [
        "Move X=10", 
        "Move Y=20", 
        "Set speed to 100"
    ],
    "gcode": [
        "G1 X10", 
        "G1 Y20", 
        "S100"
    ]
}
df = pd.DataFrame(data)
dataset = Dataset.from_pandas(df)

# Load model and tokenizer
model_name = "t5-small"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Preprocess the dataset
def preprocess_function(examples):
    inputs = examples["instruction"]
    targets = examples["gcode"]
    model_inputs = tokenizer(inputs, max_length=512, return_tensors="pt", padding=True, truncation=True)
    labels = tokenizer(targets, max_length=512, truncation=True, padding=True).input_ids
    model_inputs["labels"] = labels
    return model_inputs

tokenized_dataset = dataset.map(preprocess_function, batched=True)

# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    logging_dir="./logs",
    logging_steps=10,
)

# Define Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
)

# Fine-tune model
trainer.train()

# Save model
model.save_pretrained("./gcode_translation_model")
tokenizer.save_pretrained("./gcode_translation_model")
