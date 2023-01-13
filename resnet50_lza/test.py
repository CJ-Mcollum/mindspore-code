import os
import stat
import numpy as np
import matplotlib.pyplot as plt
import math

import mindspore.nn as nn
import mindspore.dataset as ds
import mindspore.dataset.vision.c_transforms as CV
import mindspore.dataset.transforms.c_transforms as C
from mindspore import dtype as mstype
from mindspore.train.callback import TimeMonitor, Callback
from mindspore import Model, Tensor, context, save_checkpoint, load_checkpoint, load_param_into_net
from resnet import resnet50

from read_json import read_json
from count_img_num import count_img_num

# val_data_path = r"F:\dataset\Lung-nodules\exp_nodules\train_val_test\nodules_cascade_tvt\tag_crop64_ms\nodules_ground_solid_exp_tvt_ms\val"
val_data_path = r"F:\dataset\Lung-nodules\exp_nodules\train_val_test\nodules_cascade_tvt\tag_crop64_ms\nodules_gbenign_gmali_exp_tvt_ms\val"

best_ckpt_path = "./model/tvt/gbenign_gmali_best.ckpt"
json_path = "./json/gbenign_gmali.json"


def create_dataset(data_path, batch_size=24, repeat_num=1, training=True):
    """定义数据集"""
    data_set = ds.ImageFolderDataset(data_path, num_parallel_workers=8, shuffle=True)

    image_size = [224, 224]
    mean = [0.485 * 255, 0.456 * 255, 0.406 * 255]
    std = [0.229 * 255, 0.224 * 255, 0.225 * 255]
    trans = [
        CV.Decode(),
        CV.Resize(image_size),
        CV.Normalize(mean=mean, std=std),
        CV.HWC2CHW()
    ]

    # 实现数据的map映射、批量处理和数据重复的操作
    type_cast_op = C.TypeCast(mstype.int32)
    data_set = data_set.map(operations=trans, input_columns="image", num_parallel_workers=8)
    data_set = data_set.map(operations=type_cast_op, input_columns="label", num_parallel_workers=8)
    data_set = data_set.batch(batch_size, drop_remainder=True)
    data_set = data_set.repeat(repeat_num)

    return data_set


def infer_model(best_ckpt_path, val_data_path, json_path, batch_size):
    img_sum = count_img_num(val_data_path)
    val_ds = create_dataset(val_data_path, batch_size=batch_size)

    class_name = read_json(json_path)

    num_class = len(class_name)

    net = resnet50(num_class)

    param_dict = load_checkpoint(best_ckpt_path)
    load_param_into_net(net, param_dict)
    loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
    model = Model(net, loss, metrics={"Accuracy": nn.Accuracy()})
    data = next(val_ds.create_dict_iterator())

    # print("data:", data)
    labels = data["label"].asnumpy()
    print("labels:", labels)

    # output = model.predict(Tensor(data['image']))
    # pred = np.argmax(output.asnumpy(), axis=1)

    return np.sum(labels == 0), np.sum(labels == 1)


def infer_main():
    sum_0, sum_1 = 0, 0
    image_sum = 100
    batch_size = 70
    # label0, label1 = infer_model(best_ckpt_path, val_data_path, json_path, batch_size)
    # print("label0, label1:", label0, label1)
    counts = math.ceil(image_sum / batch_size)
    print(counts)

    for i in range(counts):
        label0, label1 = infer_model(best_ckpt_path, val_data_path, json_path, batch_size)
        print("label0, label1:", label0, label1)
        sum_0 += label0
        sum_1 += label1

    print("sum_0:", sum_0)
    print("sum_1:", sum_1)


if __name__ == '__main__':
    infer_main()
