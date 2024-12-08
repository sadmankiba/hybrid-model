import torch
from torch.utils.data import DataLoader

from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset


def collate_fn(batch):
    input_ids = torch.tensor([item["input_ids"] for item in batch])
    attention_mask = torch.tensor([item["attention_mask"] for item in batch])
    labels = torch.tensor([item["label"] for item in batch])
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

device = "cuda:3"
tokenizer = AutoTokenizer.from_pretrained("state-spaces/mamba-130m-hf")
##tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")  # Replace with the actual Mamba model ID
model = AutoModelForCausalLM.from_pretrained("state-spaces/mamba-130m-hf").to(device)

dataset = load_dataset("imdb", split="test")
#tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)

tokenized_dataset = dataset.map(tokenize_function, batched=True)

dataloader = DataLoader(tokenized_dataset, batch_size=8, collate_fn=collate_fn)


# Load tokenizer and pretrained model

model.eval()  # Set model to evaluation mode
# Load IMDb dataset

model.to(device)
total_loss = 0
all_predictions = []
all_labels = []

with torch.no_grad():  # Disable gradient computation
    for idx, batch in enumerate(dataloader):
        # Move input tensors to the correct device
        input_ids = batch["input_ids"].to(model.device)
        attention_mask = batch["attention_mask"].to(model.device)
        labels = batch["labels"].to(model.device)
        print(input_ids.shape)
        print(labels.shape)
        # Forward pass
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        logits = outputs.logits
        loss = outputs.loss
        # Accumulate loss
        total_loss += loss.item()
        # Get predictions and true labels
        predictions = torch.argmax(logits, dim=-1).cpu().numpy()
        labels = labels.cpu().numpy()
        # Store predictions and labels for metric calculation
        all_predictions.extend(predictions.flatten())
        all_labels.extend(labels.flatten())
        if idx % 10 == 0:
            print(f"Processed {idx} batches")