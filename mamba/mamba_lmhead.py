from collections import namedtuple

import torch
import torch.nn as nn
from transformers import AutoTokenizer, MambaForCausalLM


class MambaTextClassification(nn.Module):
    def __init__(self, model_name, n_classes) -> None:
        super(MambaTextClassification, self).__init__() 
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
    
        self.backbone = MambaForCausalLM.from_pretrained(model_name, 
            pad_token_id=self.tokenizer.pad_token_id, output_hidden_states=True)
        d_model = self.backbone.config.hidden_size
        
        self.cls_head = nn.Linear(d_model, n_classes)

    def forward(self, input_ids, attention_mask=None, labels=None):
        output = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        backbone_logits = output.logits # (batch_size, seq_len, vocab_size)
        
        # hidden_states is a Tuple of FloatTensors. Contains hidden states of the 
        # model at each layer + normalized last hidden state. All FloatTensors has same shape 
        # (batch_size, seq_len, d_model).  
        last_hidden_states = output.hidden_states[-1]  
        mean_hidden_states = last_hidden_states.mean(dim=1) # (batch_size, d_model)
        logits = self.cls_head(mean_hidden_states) # (batch_size, n_classes)

        if labels is None:
          ClassificationOutput = namedtuple("ClassificationOutput", ["logits"])
          return ClassificationOutput(logits=logits)
        else:
          ClassificationOutput = namedtuple("ClassificationOutput", ["loss", "logits"])

          loss_fct = nn.CrossEntropyLoss()
          loss = loss_fct(logits, labels)

          return ClassificationOutput(loss=loss, logits=logits)