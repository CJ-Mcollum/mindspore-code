
import os
import stat
import numpy as np
import matplotlib.pyplot as plt

import mindspore.nn as nn
import mindspore.dataset as ds
import mindspore.dataset.vision.c_transforms as CV
import mindspore.dataset.transforms.c_transforms as C
from mindspore import dtype as mstype
from mindspore.train.callback import TimeMonitor, Callback
from mindspore import Model, Tensor, context, save_checkpoint, load_checkpoint, load_param_into_net
from resnet import resnet50

from count_img_num import count_img_num
from make_json import make_json
from read_json import read_json

from methods_set import write_txt

#加载开源数据集训练模型，冻结卷模型，删除并重新训练最后一层参数
#设置使用设备，CPU/GPU/Ascend
device_target = "Ascend"
PWD = os.path.dirname(os.path.abspath(__file__))

#数据路径
# train_data_path = 'dataset/train'
# val_data_path = 'dataset/val'
dataset_path = os.path.join(PWD, "data_nodules/ground_solid/")

train_data_path = os.path.join(dataset_path, "train")
val_data_path = os.path.join(dataset_path, "val")
test_data_path = os.path.join(dataset_path, "test")

tvt_path = (train_data_path, val_data_path, test_data_path)

json_path = "./json/{}.json".format("ground_solid_DL")
con_save_path = "./model/{}_continue_50_0.96.ckpt".format("ground_solid_DL")
trans_save_path = "./model/{}_trainsfer.ckpt".format("ground_solid_DL")
num_epochs = 50
batch_size_num = 24

train_loss = []
train_acc = []
test_acc = []


def create_dataset(data_path, batch_size=24, repeat_num=1):
    """定义数据集"""
    data_set = ds.ImageFolderDataset(data_path, num_parallel_workers=8, shuffle=True)

    # 对数据进行预处理
    image_size = [224, 224]
    mean = [0.485 * 255, 0.456 * 255, 0.406 * 255]
    std = [0.229 * 255, 0.224 * 255, 0.225 * 255]
    trans = [
        CV.Decode(),
        CV.Resize(image_size),
        CV.Normalize(mean=mean, std=std),
        CV.HWC2CHW()
    ]
    type_cast_op = C.TypeCast(mstype.int32)

    # 实现数据的map映射、批量处理和数据重复的操作
    data_set = data_set.map(operations=trans, input_columns="image", num_parallel_workers=8)
    data_set = data_set.map(operations=type_cast_op, input_columns="label", num_parallel_workers=8)
    data_set = data_set.batch(batch_size, drop_remainder=True)
    data_set = data_set.repeat(repeat_num)

    return data_set


#实例化数据集处理
train_ds = create_dataset(train_data_path)


# 模型验证
def apply_eval(eval_param):
    eval_model = eval_param['model']
    eval_ds = eval_param['dataset']
    metrics_name = eval_param['metrics_name']
    res = eval_model.eval(eval_ds)
    return res[metrics_name]

class EvalCallBack(Callback):
    """
    回调类，获取训练过程中模型的信息
    """

    def __init__(self, eval_function, eval_param_dict, infer_param_dict, num_epochs, interval=1, eval_start_epoch=1, save_best_ckpt=True,
                 ckpt_directory="./", besk_ckpt_name="best.ckpt", metrics_name="acc"):
        super(EvalCallBack, self).__init__()
        self.eval_param_dict = eval_param_dict
        self.infer_param_dict = infer_param_dict
        self.eval_function = eval_function
        self.eval_start_epoch = eval_start_epoch
        if interval < 1:
            raise ValueError("interval should >= 1.")
        self.interval = interval
        self.save_best_ckpt = save_best_ckpt
        self.best_res = 0
        self.best_epoch = 0
        self.infer_best_res = 0
        self.infer_best_epoch = 0
        if not os.path.isdir(ckpt_directory):
            os.makedirs(ckpt_directory)
        self.best_ckpt_path = os.path.join(ckpt_directory, besk_ckpt_name)
        self.metrics_name = metrics_name
        self.num_epochs = num_epochs

    # 删除ckpt文件
    def remove_ckpoint_file(self, file_name):
        os.chmod(file_name, stat.S_IWRITE)
        os.remove(file_name)

    # 每一个epoch后，打印训练集的损失值和验证集的模型精度，并保存精度最好的ckpt文件
    def epoch_end(self, run_context):
        cb_params = run_context.original_args()
        cur_epoch = cb_params.cur_epoch_num
        loss_epoch = cb_params.net_outputs
        if cur_epoch >= self.eval_start_epoch and (cur_epoch - self.eval_start_epoch) % self.interval == 0:
            res = self.eval_function(self.eval_param_dict)
            infer_res = self.eval_function(self.infer_param_dict)

            print()
            print('Epoch {}/{}'.format(cur_epoch, self.num_epochs))
            print('-' * 10)
            print('train Loss: {}'.format(loss_epoch))
            print('val Acc: {}'.format(res))
            print('val best Acc: {}'.format(self.best_res))

            print('infer Acc: {}'.format(infer_res))
            print('infer best Acc: {}'.format(self.infer_best_res))

            if res >= self.best_res:
                self.best_res = res
                self.best_epoch = cur_epoch
                # if self.save_best_ckpt:
                #     if os.path.exists(self.best_ckpt_path):
                #         self.remove_ckpoint_file(self.best_ckpt_path)
                #     save_checkpoint(cb_params.train_network, self.best_ckpt_path)

            if infer_res >= self.infer_best_res:
                self.infer_best_res = infer_res
                self.infer_best_epoch = cur_epoch
                if self.save_best_ckpt:
                    if os.path.exists(self.best_ckpt_path):
                        self.remove_ckpoint_file(self.best_ckpt_path)
                    save_checkpoint(cb_params.train_network, self.best_ckpt_path)

            train_loss.append(loss_epoch)
            train_acc.append(res)
            test_acc.append(infer_res)

    # 训练结束后，打印最好的精度和对应的epoch
    def end(self, run_context):
        print("End training, the best val {0} is: {1}, the best infer {0} is: {3}, the best val epoch is {2}, "
              "the best val epoch is {4}".format(self.metrics_name,
                                                 self.best_res,
                                                 self.best_epoch,
                                                 self.infer_best_res,
                                                 self.infer_best_epoch), flush=True)
        # print("End training, the best {0} is: {1}, the best {0} epoch is {2}".format(self.metrics_name,
        #                                                                              self.best_res,
        #                                                                              self.best_epoch), flush=True)


def filter_checkpoint_parameter_by_list(origin_dict, param_filter):
    for key in list(origin_dict.keys()):
        for name in param_filter:
            if name in key:
                print("Delete parameter from checkpoint: ", key)
                del origin_dict[key]
                break


def train_transfer_main(tvt_path, num_epochs, json_path, con_save_path, trans_save_path, batch_size, device_target):
    context.set_context(mode=context.GRAPH_MODE, device_target=device_target)

    train_data_path, val_data_path, test_data_path = tvt_path

    # 读取 Json文件
    class_name = read_json(json_path)
    print("\nlabels:", class_name)
    num_class = len(class_name)
    print("num_class:", num_class)

    # 定义网络
    net = resnet50(num_class)
    # num_epochs=10

    # 加载预训练模型
    param_dict = load_checkpoint(con_save_path)
    # param_dict = load_checkpoint('resnet50.ckpt')

    # 获取最后一层参数的名字
    filter_list = [x.name for x in net.end_point.get_parameters()]

    # 删除预训练模型最后一层的参数
    filter_checkpoint_parameter_by_list(param_dict, filter_list)

    # 给网络加载参数
    load_param_into_net(net, param_dict)

    # 冻结除最后一层外的所有参数
    for param in net.get_parameters():
        if param.name not in ["end_point.weight","end_point.bias"]:
             param.requires_grad = False

    # 定义优化器和损失函数
    opt = nn.Momentum(params=net.trainable_params(), learning_rate=0.01, momentum=0.9)
    loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')

    # 实例化模型
    model = Model(net, loss,opt,metrics={"Accuracy":nn.Accuracy()})

    print(dataset_path)
    print("using {} images for training, {} images for validation, {} images for inference.".format(
        count_img_num(train_data_path), count_img_num(val_data_path), count_img_num(test_data_path)
    ))

    # 设置batch_size大小
    train_batch_size, val_batch_size, test_batch_size = batch_size, batch_size, batch_size
    print(train_batch_size, val_batch_size, test_batch_size)

    # 判断batch_size大小
    if batch_size > count_img_num(train_data_path):
        train_batch_size = count_img_num(train_data_path)

    if batch_size > count_img_num(val_data_path):
        val_batch_size = count_img_num(val_data_path)

    if batch_size > count_img_num(test_data_path):
        test_batch_size = count_img_num(test_data_path)

    print(train_batch_size, val_batch_size, test_batch_size)

    # 加载训练和验证数据集
    train_ds = create_dataset(train_data_path, batch_size=train_batch_size)
    val_ds = create_dataset(val_data_path, batch_size=val_batch_size)
    test_ds = create_dataset(test_data_path, batch_size=test_batch_size)

    # 实例化回调类
    eval_param_dict = {"model":model,"dataset":val_ds,"metrics_name":"Accuracy"}
    infer_param_dict = {"model": model,"dataset": test_ds, "metrics_name": "Accuracy"}

    # eval_cb = EvalCallBack(apply_eval, eval_param_dict,)
    eval_cb = EvalCallBack(apply_eval, eval_param_dict, infer_param_dict, num_epochs, besk_ckpt_name=trans_save_path)

    print("\n* trainsfer training start\n")
    # 模型训练
    model.train(num_epochs,train_ds, callbacks=[eval_cb, TimeMonitor()], dataset_sink_mode=False)
    print()
    print("Transfer training has been completed")
    print("train set:", train_data_path)
    print("val set:", val_data_path)
    print("test set:", test_data_path)
    print()


if __name__ == '__main__':
    train_transfer_main(tvt_path, num_epochs, json_path, con_save_path, trans_save_path, batch_size_num, device_target)
