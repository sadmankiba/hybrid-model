import torch
from datasets import load_dataset

def preprocess_squad():
    pass

class SquadPrerprocessing:
    def __init__(self, tokenizer, max_length):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def get_squad_causal_dataset(self, tokenizer):
        """
        Convert SQUAD dataset into question-answer input-output pairs
        
        Each output contains one or more answer words
        """
        squad_dataset_id = "rajpurkar/squad_v2"
        squad_train_dataset = load_dataset(squad_dataset_id, split="train")
        squad_val_dataset = load_dataset(squad_dataset_id, split="validation")
        
        train_items = []
        for item in squad_train_dataset:
            chunked_items = self._chunk_answer(item)
            train_items.extend(chunked_items)

        train_items = self._chunk_answer(squad_train_dataset, tokenizer, self.max_length)
        val_items = self._chunk_answer(squad_val_dataset, tokenizer, self.max_length)
        return train_items, val_items
    
    def _chunk_answer(self, item):
        items = []
        
        context = item['context']
        question = item['question']
        answer = item['answers']['text'][0]
        text = f"Context: {context} Question: {question} Answer: {answer}"

        # Pads only if needed (multiple examples in a batch), otherwise does not 
        # So, add a end-of-sequence token to the end of the input
        encoding = self.tokenizer(text, padding=True, truncation=True, max_length=self.max_length, return_tensors='pt')
        encoding = {k: v.squeeze() for k, v in encoding.items()}
        encoding["input_ids"] = torch.concatenate((encoding["input_ids"], torch.tensor([self.tokenizer.eos_token_id])))
        encoding["attention_mask"] = torch.concatenate((encoding["attention_mask"], torch.tensor([1]))) 
        print("encoding", encoding)
        
        answer_template = " Answer:"
        ans_start_pos = self._get_answer_tokens_end_pos(encoding, answer_template) + 1
        cur_pos = ans_start_pos
        
        # Chunk up answer into multiple examples
        input_ids_orig = encoding["input_ids"]
        attention_mask_orig = encoding["attention_mask"]
        while cur_pos < len(encoding["input_ids"]):
            input_ids = input_ids_orig[:cur_pos]
            print("input_ids", input_ids)
            input_ids = torch.concatenate((input_ids,
                torch.tensor([self.tokenizer.eos_token_id] * (self.max_length - len(input_ids)))))
            print("input_ids", input_ids)
            
            attention_mask = attention_mask_orig[:cur_pos]
            print("attention_mask", attention_mask)
            attention_mask = torch.concatenate((attention_mask,
                torch.tensor([0] * (self.max_length - len(attention_mask)))))
            print("attention_mask", attention_mask)
            
            labels = input_ids_orig[1:cur_pos+1].clone()
            print("labels", labels)
            labels[:ans_start_pos - 1] = -100
            labels = torch.concatenate((labels, 
                torch.tensor([-100] * (self.max_length - len(labels))))) 
            print("labels", labels)
            
            items.append({
                'context': context,
                'question': question,
                'answer': answer,
                'token_ids': input_ids,
                'attention_mask': attention_mask,
                'labels': labels,
            })
            
            cur_pos += 1
    
    def _get_answer_tokens_end_pos(self, encoding, answer_template):
        answer_encoding = self.tokenizer(answer_template, padding=False, 
            truncation=True, max_length=self.max_length, 
            return_tensors='pt').input_ids[0]
        
        input_ids = encoding['input_ids']
        for i in range(len(input_ids)):
            if torch.equal(input_ids[i:i+len(answer_encoding)], answer_encoding):
                return i + len(answer_encoding) - 1
        
        return -1
        
    
        