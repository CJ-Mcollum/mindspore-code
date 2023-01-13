import os

dataset_path = r"./dataset_hl"


def count_img_num(dataset_path):
    sum = 0
    for root, dirs, files in os.walk(dataset_path):
        for file in files:
            if file.endswith('.jpg') or file.endswith('.jpeg') or file.endswith('.JPG')\
                    or file.endswith('.png') or file.endswith('.PNG'):
                sum += 1

    return sum


if __name__ == '__main__':
    print(count_img_num(dataset_path))
