import os
import json

dataset_path = "./dataset/train"
save_json = "./json/json1.json"


def make_json(dataset_path, save_json):
    labels = os.listdir(dataset_path)
    label_dict = {x:labels[x] for x in range(len(labels))}
    json_str = json.dumps(label_dict, indent=4)
    with open(save_json, "w", encoding='utf-8') as f:
        f.write(json_str)


if __name__ == '__main__':
    make_json(dataset_path, save_json)
