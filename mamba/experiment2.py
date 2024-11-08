from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import evaluate
import torch
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
device = "cuda:3"
# Load tokenizer and pretrained model
tokenizer = AutoTokenizer.from_pretrained("state-spaces/mamba-130m-hf")
##tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")  # Replace with the actual Mamba model ID
model = AutoModelForCausalLM.from_pretrained("state-spaces/mamba-130m-hf").to(device)
model.eval()  # Set model to evaluation mode
# Load IMDb dataset

dataset = load_dataset("imdb", split="test")

# Tokenize the dataset
print(model.device)
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)
tokenized_dataset = dataset.map(tokenize_function, batched=True)
# Set up data loader with collate_fn to ensure tensors are returned
def collate_fn(batch):
    input_ids = torch.tensor([item["input_ids"] for item in batch])
    attention_mask = torch.tensor([item["attention_mask"] for item in batch])
    labels = torch.tensor([item["label"] for item in batch])
    return {"input_ids": input_ids, "attention_mask":attention_mask,  "labels": labels}
dataloader = DataLoader(tokenized_dataset, batch_size=8, collate_fn=collate_fn)
print(tokenized_dataset[0])

# Define metrics
accuracy_metric = evaluate.load("accuracy")
f1_metric = evaluate.load("f1")
# Initialize lists for storing results
all_predictions = []
all_labels = []
total_loss = 0
# Evaluate model
with torch.no_grad():  # Disable gradient computation
    for idx, batch in enumerate(dataloader):
        # Move input tensors to the correct device
        input_ids = batch["input_ids"].to(model.device)
        attention_mask = batch["attention_mask"].to(model.device)
        labels = batch["labels"].to(model.device)
        print(input_ids.shape)
    
        print(labels.shape)
        # Forward pass
        outputs = model(input_ids=input_ids, attention_mask = attention_mask, labels=labels)
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
        if idx %10==0:
            print(f"Batch {idx+1}")
        
        break
print(all_predictions)
print(len(all_predictions))
print("===============")
print(len(all_labels))
print(all_labels)

# Calculate average loss
avg_loss = total_loss / len(dataloader)
# Calculate accuracy and F1 score
accuracy = accuracy_metric.compute(predictions=all_predictions, references=all_labels)["accuracy"]
f1 = f1_metric.compute(predictions=all_predictions, references=all_labels, average="weighted")["f1"]
# Print results
print(f"Average Loss: {avg_loss}")
print(f"Accuracy: {accuracy}")
print(f"F1 Score: {f1}")