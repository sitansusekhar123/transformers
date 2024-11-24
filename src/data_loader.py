from torch.utils.data import Dataset
from data_processing import indexed_tokens_per_text
import json
import torch

def collate_fn(batch):
    # Separate x and y
    x_batch, y_batch = zip(*batch)
    
    # Find the maximum lengths
    max_len_x = max(len(x) for x in x_batch)
    max_len_y = max(len(y) for y in y_batch)
    
    # Pad x and y
    x_padded = torch.stack([torch.nn.functional.pad(x, (0, max_len_x - len(x)), value=0) for x in x_batch])
    y_padded = torch.stack([torch.nn.functional.pad(y, (0, max_len_y - len(y)), value=0) for y in y_batch])
    
    return x_padded, y_padded

class DatasetLanguage(Dataset):
    def __init__(self, data_path: str, x_vocab: str, y_vocab: str):
        self.data_path = data_path
        with open(data_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
            self.x_data = self.data['x']
            self.y_data = self.data['y']
            
        # load vocab map
        with open(x_vocab, 'r', encoding='utf-8') as f:
            self.x_vocab = json.load(f)
        
        with open(y_vocab, 'r', encoding='utf-8') as f:
            self.y_vocab = json.load(f)
                
                
    def __len__(self):
        return len(self.x_data)
    
    def __getitem__(self, idx):
        x_data = self.x_data[idx]
        x_data = indexed_tokens_per_text(x_data, self.x_vocab)
        
        y_data = self.y_data[idx]
        y_data = indexed_tokens_per_text(y_data, self.y_vocab)
        
        return torch.tensor(x_data), torch.tensor(y_data)