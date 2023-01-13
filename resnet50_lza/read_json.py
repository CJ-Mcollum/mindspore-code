import os
import json

json_path = "./json/json1.json"


def read_json(json_path):
    json_file = open(json_path, "r")
    class_indict = json.load(json_file)
    # print(class_indict)
    class_name = {int(k): d for k, d in class_indict.items()}
    # print(class_name)

    return class_name


if __name__ == '__main__':
    class_name = read_json(json_path)
    print(class_name)