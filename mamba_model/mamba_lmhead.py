from collections import namedtuple
from typing import Union

import torch
import torch.nn as nn
from transformers import AutoTokenizer, MambaModel, MambaConfig


class MambaTextClassification(nn.Module):
    def __init__(self, tokenizer_name: str, model_config,num_labels = None) -> None:    #model: Union[str, nn.Module]
        super(MambaTextClassification, self).__init__() 
        
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        if type(model_config) == str:
          self.backbone = MambaModel.from_pretrained(model_config)
        else:
          self.backbone = MambaModel(MambaConfig(model_config.vocab_size, model_config.hidden_size, num_hidden_layers = model_config.num_mamba_layers, pad_token_id=self.tokenizer.pad_token_id))    

        d_model = self.backbone.config.hidden_size
        self.cls_head = nn.Linear(d_model, num_labels if num_labels else model_config.num_labels )

    def forward(self, input_ids, attention_mask=None, labels=None):
        output = self.backbone(input_ids=input_ids, cache_position=attention_mask, output_hidden_states=False)
        # backbone_logits = output.logits # (batch_size, seq_len, vocab_size)
        
        # hidden_states is a Tuple of FloatTensors. Contains hidden states of the 
        # model at each layer + normalized last hidden state. All FloatTensors has same shape 
        # (batch_size, seq_len, d_model).  
        # print("output", output)
        # print("output fields:")
        # for field in output._fields:
        #   print("field:", field)
            
        # exit()
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
