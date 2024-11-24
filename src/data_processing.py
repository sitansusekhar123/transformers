
import json

def split_and_add_special(text: str) -> list[str]:
    tokens = text.split()
    tokens = ['sos'] + tokens + ['eos']
    return tokens

def flatten_and_unique(text_list: list[str]):
    tokens_list = [split_and_add_special(t) for t in text_list]
    vocabulary = set(token for tokens in tokens_list for token in tokens)
    
    # Map tokens to integers
    vocab_to_index = {token: idx for idx, token in enumerate(vocabulary)}
    return vocab_to_index

def indexed_tokens(text_list: list[str]):
    vocab_map = flatten_and_unique(text_list)
    tokens_list = [split_and_add_special(t) for t in text_list]
    indexed_tokens = [[vocab_map[token] for token in tokens] for tokens in tokens_list]
    return indexed_tokens, vocab_map


def indexed_tokens_per_text(text: str, vocab_map):
    tokens_list = split_and_add_special(text)
    indexed_tokens = [vocab_map[token] for token in tokens_list]
    return indexed_tokens


def data_split(x, y, split_pct: dict):
    length = len(x)
    
    train_index_start = 0
    train_index_end = int(split_pct['train'] * length)
    x_train, y_train = x[train_index_start: train_index_end], y[train_index_start: train_index_end]
    
    val_index_start = train_index_end
    val_index_end = int(split_pct['validation'] * length + train_index_end)
    x_val, y_val = x[val_index_start: val_index_end], y[val_index_start: val_index_end]
    
    test_index_start = val_index_end
    test_index_end = int(split_pct['test'] * length + val_index_end)
    x_test, y_test = x[test_index_start: test_index_end], y[test_index_start: test_index_end]
    
    return {
        "train":
            {
                "x": x_train,
                "y": y_train
            },
        "test": 
            {
                "x": x_test,
                "y": y_test
            },
        "validation":
            {
                "x": x_val,
                "y": y_val
            }
    }
    