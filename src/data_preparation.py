import os
import json
from data_processing import flatten_and_unique

def store_mapping(data_path: str):
    with open(data_path, 'r', encoding='utf-8') as f:
        data = f.readlines()
        
    data_map = flatten_and_unique(data)
    
    directory_name = os.path.dirname(data_path)
    file_name = f'{os.path.basename(data_path).split('.')[0]}.json'
    
    with open(os.path.join(directory_name, file_name), 'w', encoding='utf-8') as f:
        json.dump(data_map, f, ensure_ascii=False)
        
    return os.path.join(directory_name, file_name)

def store_split_data(data, base_path):
    for key, value in data.items():
        with open(f'{base_path}/{key}.json', 'w', encoding='utf-8') as f:
            json.dump(value, f, ensure_ascii=False, indent=4)