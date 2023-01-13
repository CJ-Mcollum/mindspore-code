import os
import json

from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import onnxruntime as ort

img_path = "./pic/I4400000_crop.png"
json_path = "./cascade_json/class_indices_ground_solid.json"
weights_path = "./lung_resNet34_ground_benign_mali_enhancement_light2_0.9137.onnx"


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


def softmax(x):
    x = x.reshape(-1)
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


def postprocess(result):
    return softmax(np.array(result)).tolist()


def onnx_main(img_path, json_path, weights_path):
    # load image
    # img_path = "./pic/ground_benign1.png"  # 正确  正确
    # img_path = "./pic/ground_mali_groMain1.png"  # 正确  正确
    # img_path = "./pic/ground_mali_solidMain_goodPro2.png"  # 正确  正确
    # img_path = "./pic/ground_mali_solidMain_poorPro2.png"  # 正确  错误
    # img_path = "./pic/solid_benign3.png"  # 正确
    # img_path = "./pic/solid_malignant1.png"  # 正确
    # img_path = "./pic/ground_benign1.png"
    # img_path = "./pic/ground_mali_groMain2.png"

    assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
    # img = Image.open(img_path)
    img = Image.open(img_path).convert('RGB')
    img = img.resize((224, 224))
    img = np.array(img)

    # plt.imshow(img)
    # [N, C, H, W]
    # expand batch dimension
    img = np.expand_dims(img, axis=0).transpose(0, 3, 1, 2)
    # print(img.shape)

    # read class_indict
    # json_path = './json/class_indices_ground_subclassify.json'
    assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)

    json_file = open(json_path, "r")
    class_indict = json.load(json_file)
    # print(len(class_indict))
    # create model
    num_classes = len(class_indict)
    # print("The classification number is ", num_classes)

    # load model weights
    # weights_path = "./dichotomy_model/lung_resNet34_ground_solid_0.9404.pth"
    # weights_path = "./dichotomy_model/lung_resNet34_ground_subclassify_0.6455.pth"

    assert os.path.exists(weights_path), "file: '{}' dose not exist.".format(weights_path)

    # prediction
    session = ort.InferenceSession(weights_path)
    # compute ONNX Runtime output prediction
    inputs = {session.get_inputs()[0].name: img.astype('float32')}

    # logits = session.run(None, inputs)[0]
    onnx_out = session.run(None, inputs)
    print("onnx_out:", onnx_out)
    logits = onnx_out[0]

    res = postprocess(logits) # 后处理 softmax
    print("res:", res)
    # print(res[0], 1-res[0])
    # print(res[1])
    idx = np.argmax(res)
    # print("idx:", idx)

    print("class_indict:", class_indict)
    # result=class_indict[idx]
    # print(result)

    print("onnx weights", logits)
    print("onnx prediction", logits.argmax(axis=1)[0])
    print("onnx prediction", logits.argmax(axis=1))

    # plt.show()
    return class_indict[str(logits.argmax(axis=1)[0])]


if __name__ == '__main__':
    onnx_main(img_path, json_path, weights_path)