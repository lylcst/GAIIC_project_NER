from torch.utils.data import Dataset, DataLoader
import torch
import json


class DataReader(Dataset):

    def __init__(self, data_path):
        super(DataReader, self).__init__()
        self.data_path = data_path
        # (self.input_ids, self.attention_mask, self.token_type_ids,
        #  self.masked_positions, self.masked_lm_ids, self.next_sentence_labels) = self.process()
        (self.input_ids, self.attention_mask, self.token_type_ids,
         self.masked_positions, self.masked_lm_ids, self.masked_lm_weights) = self.process()
    
    def __getitem__(self, idx):
        return self.input_ids[idx], self.attention_mask[idx], self.token_type_ids[idx], \
         self.masked_positions[idx], self.masked_lm_ids[idx], self.masked_lm_weights[idx]
        #  self.next_sentence_labels[idx]

    def __len__(self):
        return len(self.input_ids)

    def process(self):
        with open(self.data_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert data
        input_ids = []
        attention_mask = []
        token_type_ids = []
        masked_positions = []
        masked_lm_ids = []
        masked_lm_weights = []
        next_sentence_label = []

        for d in data:
            input_ids.append(d['input_ids'])
            attention_mask.append(d['input_mask'])
            token_type_ids.append(d['segment_ids'])
            masked_positions.append(d['masked_lm_positions'])
            masked_lm_ids.append(d['masked_lm_ids'])
            masked_lm_weights.append(d["masked_lm_weights"])

            # next_sentence_label.append(d['next_sentence_labels'])
        
        return torch.as_tensor(input_ids), \
               torch.as_tensor(attention_mask), \
               torch.as_tensor(token_type_ids), \
               torch.as_tensor(masked_positions), \
               torch.as_tensor(masked_lm_ids), \
               torch.as_tensor(masked_lm_weights)
            #    torch.as_tensor(next_sentence_label)

