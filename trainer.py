import random
import math
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.data.dataset
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import classification_report, f1_score, recall_score, accuracy_score

from datasets import load_dataset
from mad.gen_data import generate_data

TQDM_DISABLE = False

class HybridDataset(Dataset):
    def __init__(self, dataset, args, tokenizer_id='EleutherAI/gpt-neo-125M', n_samples=None):
        self.dataset = dataset
        self.p = args
        self.tokenizer =  AutoTokenizer.from_pretrained(tokenizer_id)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        ele = self.dataset[idx]
        return ele

    def pad_data(self, data):
        sents = [x["text"] for x in data]
        labels = [x["label"] for x in data]
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        encoding = self.tokenizer(sents, return_tensors='pt', padding=True, truncation=True)
        token_ids = torch.LongTensor(encoding['input_ids'])
        attention_mask = torch.LongTensor(encoding['attention_mask'])
        labels = torch.LongTensor(labels)

        return token_ids, attention_mask, labels, sents

    def collate_fn(self, all_data):
        """
        Args:
            all_data (List[Dict]): list of data points, each data point is a dictionary
        """
        all_data.sort(key=lambda x: len(x['text']), reverse=True)  # sort by number of chars
    
        batches = []
        num_batches = int(np.ceil(len(all_data) / self.p.batch_size))

        for i in range(num_batches):
            start_idx = i * self.p.batch_size
            data = all_data[start_idx: start_idx + self.p.batch_size]

            token_ids, attention_mask, labels, sents = self.pad_data(data)
            batches.append({
                'token_ids': token_ids,
                'attention_mask': attention_mask,
                'labels': labels,
                'sents': sents,
            })

        return batches

    

class Trainer:
    @staticmethod
    def model_eval(dataloader, model, device):
        model.eval() # switch to eval model, will turn off randomness like dropout
        y_true = []
        y_pred = []
        sents = []
        for step, batch in enumerate(tqdm(dataloader, desc=f'eval', disable=TQDM_DISABLE)):
            b_ids, b_mask, b_labels, b_sents = batch[0]['token_ids'], \
                    batch[0]['attention_mask'], batch[0]['labels'], batch[0]['sents']

            b_ids = b_ids.to(device)
            b_mask = b_mask.to(device)

            with torch.no_grad():
                output = model(input_ids=b_ids, attention_mask=b_mask)
                logits = output.logits.detach().cpu().numpy()
           
            preds = np.argmax(logits, axis=1).flatten()

            b_labels = b_labels.flatten()
            y_true.extend(b_labels)
            y_pred.extend(preds)
            sents.extend(b_sents)
            
            if step % 5 == 0:
                # Can cut down the memory usage to 1/3rd, but slows down evaluation
                torch.cuda.empty_cache()  # Releases all unoccupied cached memory currently held by the caching allocator
                torch.cuda.synchronize()  # Wait for all kernels in all streams on a CUDA device to complete

        f1 = f1_score(y_true, y_pred, average='macro')
        acc = accuracy_score(y_true, y_pred)

        return acc, f1, y_pred, y_true, sents
    
    
    @staticmethod
    def train(model, tokenizer_id, dataset_name, param_list, args):
        device  = torch.device('cuda') if args.use_gpu else torch.device('cpu') 
        #### Load data
        # create the data and its corresponding datasets and dataloader
        print("device:", device)
        dataset = load_dataset(dataset_name, split="train")
        
        train_data, dev_data = random_split(dataset, [0.8, 0.2])
        if args.train_size > 0:
            train_data = torch.utils.data.dataset.Subset(train_data, range(args.train_size))
        
        if args.eval_size > 0:
            dev_data = torch.utils.data.dataset.Subset(dev_data, range(args.eval_size))
        
        train_dataset = HybridDataset(train_data, args, tokenizer_id)
        dev_dataset = HybridDataset(dev_data, args, tokenizer_id)

        train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size,
                                    collate_fn=train_dataset.collate_fn)
        dev_dataloader = DataLoader(dev_dataset, shuffle=False, batch_size=args.batch_size,
                                    collate_fn=dev_dataset.collate_fn)

        #### Init model
        config = {'hidden_dropout_prob': args.hidden_dropout_prob,
                'num_labels': args.num_labels,
                'hidden_size': 768,
                'data_dir': '.',
                'option': args.option}

        config = SimpleNamespace(**config)

        # initialize the Senetence Classification Model
        use_amp = True   # Mixed Precision Training reduces memory usage
        scaler = torch.amp.GradScaler(enabled=use_amp)
        model = model.to(device)
        optimizer = optim.AdamW(param_list, lr=args.lr, weight_decay=args.weight_decay)
        ## run for the specified number of epochs
        best_dev_acc = 0
        print("==Started training====")
        for epoch in range(math.ceil(args.epochs)):
            model.train()
            train_loss = 0
            num_batches = 0
            print("len of train_dataloader:", len(train_dataloader))
            for step, batch in enumerate(tqdm(train_dataloader, desc=f'train-{epoch}', disable=TQDM_DISABLE)):
                b_ids, b_mask, b_labels, b_sents = batch[0]['token_ids'], batch[0]['attention_mask'], batch[0]['labels'], batch[0]['sents']

                b_ids = b_ids.to(device)
                b_mask = b_mask.to(device)
                b_labels = b_labels.to(device)
                
                with torch.autocast(device_type=str(device), dtype=torch.float16, enabled=use_amp):
                    output = model(input_ids=b_ids, attention_mask=b_mask) # For GPT-Neo, output type SequenceClassifierOutputWithPast
                    logits = output.logits # Classification: (batch_size, n_classes) # LM: (batch_size, vocab_size)
                    loss = F.nll_loss(F.log_softmax(logits, dim=1), b_labels, reduction='mean')

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                
                # Scaler is used instead of these steps
                # loss.backward()
                # optimizer.step()
                
                optimizer.zero_grad()

                train_loss += loss.item()
                num_batches += 1
                
                if args.log_interval > 0 and step % args.log_interval == 0:
                    dev_acc, dev_f1, *_ = Trainer.model_eval(dev_dataloader, model, device)
                    print(f"epoch {epoch}, step {step}: train loss :: {loss.item() :.3f}, dev acc :: {dev_acc :.3f}")

                if (epoch + step / len(train_dataloader)) >= args.epochs:
                    break

            train_loss = train_loss / (num_batches)

            train_acc, train_f1, *_ = Trainer.model_eval(train_dataloader, model, device)
            dev_acc, dev_f1, *_ = Trainer.model_eval(dev_dataloader, model, device)

            if dev_acc > best_dev_acc:
                best_dev_acc = dev_acc
                Trainer.save_model(model, optimizer, args, config, args.filepath)

            print(f"epoch {epoch}: train loss :: {train_loss :.3f}, train acc :: {train_acc :.3f}, dev acc :: {dev_acc :.3f}")

    @staticmethod
    def train_mad(model, config):
        device = torch.device('cuda') if config.use_gpu else torch.device('cpu')
        data = generate_data(
            instance_fn=config.instance_fn,
            instance_fn_kwargs=config.instance_fn_kwargs,
            train_data_path=config.train_dataset_path,
            test_data_path=config.test_dataset_path,
            num_train_examples=config.num_train_examples,
            num_test_examples=config.num_test_examples,
        )
        print("Generated data")
        
        # no padding. So, no attention_mask required.
        train_dl = DataLoader(dataset=data['train'], batch_size=config.batch_size, shuffle=True)
        test_dl = DataLoader(dataset=data['test'], batch_size=config.batch_size, shuffle=False)
        
        model = model.to(device)
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        for epoch in range(int(config.epochs)):
            model.train()
            train_loss = 0
            for step, batch in enumerate(tqdm(train_dl, desc=f'train-{epoch}', disable=TQDM_DISABLE)):
                inputs, targets = batch
                inputs, targets = inputs.to(model.device), targets.to(model.device)
                
                optimizer.zero_grad()
                output = model(input_ids=inputs)
                loss = F.cross_entropy(output.logits.view(-1, 
                            output.logits.size(-1)), targets.view(-1), ignore_index=-100)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                
                train_loss += loss.item()
                
                if config.log_interval > 0 and step % config.log_interval == 0:
                    print(f"epoch {epoch}, step {step}: train loss :: {loss.item() :.3f}")
            
            train_loss = train_loss / len(train_dl)
            
            eval_loss, eval_acc = Trainer.eval_mad(model, test_dl)
             
            print("epoch:", epoch, f"train loss: {train_loss:.3f}, eval loss: {eval_loss:.3f}, eval acc: {eval_acc:.3f}")
       
    @staticmethod 
    def eval_mad(model, eval_dl):
        model.eval()
        test_loss = 0
        total_correct = 0
        total_items = 0
        
        print_samples = True
        for step, batch in enumerate(eval_dl):
            inputs, targets = batch
            inputs, targets = inputs.to(model.device), targets.to(model.device)
            with torch.no_grad():
                output = model(input_ids=inputs)
                loss = F.cross_entropy(output.logits.view(-1, 
                            output.logits.size(-1)), targets.view(-1), ignore_index=-100)
                test_loss += loss
            
            predicted = output.logits.argmax(dim=2)
            if print_samples and step % 10 == 0:
                print("inputs:", inputs)
                print("targets:", targets)
                print("predicted:", predicted)
            
            # output.logits shape: (batch_size, seq_len, vocab_size)    
            valid_targets = (targets != -100)
            correct = valid_targets & (targets == predicted)
            total_correct += correct.sum().item()
            total_items += valid_targets.sum().item()
        
        test_loss = test_loss / len(eval_dl)    
        acc = total_correct / total_items
        
        return test_loss, acc

    @staticmethod
    def _shift_targets_right(targets):    
        """
        Add a dummy token at the beginning of targets, because they are 
           shifted internally in HF model
        
        Args:
            targets: tensor of shape (batch_size, seq_len)
        """
        dummy_token = torch.full((targets.size(0), 1), -100, dtype=targets.dtype, device=targets.device)
        targets = torch.cat([dummy_token, targets], dim=1)
        return targets
    
    @staticmethod
    def save_model(model, optimizer, args, config, filepath):
        save_info = {
            'model': model.state_dict(),
            'optim': optimizer.state_dict(),
            'args': args,
            'model_config': config,
            'system_rng': random.getstate(),
            'numpy_rng': np.random.get_state(),
            'torch_rng': torch.random.get_rng_state(),
        }

        torch.save(save_info, filepath)
        print(f"save the model to {filepath}")
    
    def test(self,args):
        with torch.no_grad():
            device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
            saved = torch.load(args.filepath)
            config = saved['model_config']
            model = model
            model.load_state_dict(saved['model'])
            model = model.to(device)
            print(f"load model from {args.filepath}")
            dev_data = create_data(args.dev, 'valid')
            dev_dataset = HybridDataset(dev_data, args)
            dev_dataloader = DataLoader(dev_dataset, shuffle=False, batch_size=args.batch_size, collate_fn=dev_dataset.collate_fn)

            test_data = create_data(args.test, 'test')
            #test_dataset = ----
            test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=args.batch_size, collate_fn=test_dataset.collate_fn)

            dev_acc, dev_f1, dev_pred, dev_true, dev_sents = model_eval(dev_dataloader, model, device)
            test_acc, test_f1, test_pred, test_true, test_sents = model_eval(test_dataloader, model, device)

            with open(args.dev_out, "w+") as f:
                print(f"dev acc :: {dev_acc :.3f}")
                for s, t, p in zip(dev_sents, dev_true, dev_pred):
                    f.write(f"{s} ||| {t} ||| {p}\n")

            with open(args.test_out, "w+") as f:
                print(f"test acc :: {test_acc :.3f}")
                for s, t, p in zip(test_sents, test_true, test_pred):
                    f.write(f"{s} ||| {t} ||| {p}\n")
