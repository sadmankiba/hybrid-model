from typing import Optional

import torch
from tqdm import tqdm
from torch.utils.data import Dataset

from datasets import load_dataset

def preprocess_squad():
    pass

class SquadDataset(Dataset):
    """
    Dataloader learns the dataset length from len(). 
    It gets an item using indexing and passes batches items
    to collate_fn to convert them into a batch. 
    """
    def __init__(self, tokenizer, max_length, split: str, num_samples:Optional[int]=None):
        """_summary_

        Args:
            tokenizer (_type_): _description_
            max_length (_type_): _description_
            split: 'train' or 'validation'
        """
        squadprep = SquadPrerprocessing(tokenizer, max_length, split=split, num_samples=num_samples)
        self.prepd_items = squadprep.get_squad_causal_dataset()
    
    def __len__(self):
        return len(self.prepd_items)
    
    def __getitem__(self, idx):
        """Return an item from the base dataset"""
        return self.prepd_items[idx]


class SquadPrerprocessing:
    def __init__(self, tokenizer, max_length, split='train', num_samples=None):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.num_samples = num_samples
        self.split = split

    def get_squad_causal_dataset(self):
        """
        Convert SQUAD dataset into question-answer input-output pairs
        
        Each output contains one or more answer words
        """
        squad_dataset_id = "rajpurkar/squad_v2"
        squad_dataset = load_dataset(squad_dataset_id, split=self.split)

        items = []
        for item in tqdm(squad_dataset, desc=f"Processing {self.split} dataset"):
            chunked_items = self._chunk_answer(item)
            items.extend(chunked_items)
            if self.num_samples and len(items) >= self.num_samples:
                items = items[:self.num_samples]
                break

        return items
    
    def _chunk_answer(self, item):
        items = []
        
        if not item['answers']['text']:
            return [] # evaluation metrics require a non-empty answer (reference)
        
        context = item['context']
        question = item['question']
        answer = item['answers']['text'][0] if item['answers']['text'] else ""
        text = f"Context: {context} Question: {question} Answer: {answer}"

        # padding=True pads only if needed (multiple examples in a batch), otherwise does not 
        # So, add a end-of-sequence token to the end of the input
        encoding = self.tokenizer(text, padding=False, truncation=True, max_length=self.max_length, return_tensors='pt')
        encoding = {k: v.squeeze() for k, v in encoding.items()}
        encoding["input_ids"] = torch.concatenate((encoding["input_ids"], torch.tensor([self.tokenizer.eos_token_id])))
        encoding["attention_mask"] = torch.concatenate((encoding["attention_mask"], torch.tensor([1]))) 
        
        answer_template = " Answer:"
        ans_start_pos = self._get_answer_template_end_pos(encoding, answer_template) + 1
        if ans_start_pos == 0:
            return [] # Answer template not found
        
        cur_pos = ans_start_pos
        
        # Chunk up answer into multiple examples
        input_ids_orig = encoding["input_ids"]
        attention_mask_orig = encoding["attention_mask"]
        while cur_pos < len(encoding["input_ids"]):
            input_ids = input_ids_orig[:cur_pos]
            input_ids = torch.concatenate((
                input_ids,
                torch.tensor([self.tokenizer.eos_token_id] * (self.max_length - len(input_ids)), dtype=torch.long)
            ))
            
            attention_mask = attention_mask_orig[:cur_pos]
            attention_mask = torch.concatenate((
                attention_mask,
                torch.tensor([0] * (self.max_length - len(attention_mask)), dtype=torch.long)
            ))
            
            labels = input_ids_orig[1:cur_pos+1].clone()
            if self.split == 'validation':
                labels[:ans_start_pos - 1] = -100
            
            labels = torch.concatenate((
                labels, 
                torch.tensor([-100] * (self.max_length - len(labels)), dtype=torch.long)
            )) 
            
            items.append({
                'context': context,
                'question': question,
                'answer': answer,
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'labels': labels,
            })
            
            cur_pos += 1
            
        return items
    
    def _get_answer_template_end_pos(self, encoding, answer_template):
        answer_encoding = self.tokenizer(answer_template, padding=False, 
            truncation=True, max_length=self.max_length, 
            return_tensors='pt').input_ids[0]
        
        input_ids = encoding['input_ids']
        for i in range(len(input_ids)):
            if torch.equal(input_ids[i:i+len(answer_encoding)], answer_encoding):
                return i + len(answer_encoding) - 1
        
        return -1
        
    
        