import numpy as np
import torch
from torchvision import models
from torch import nn
from PIL import Image
import io
import base64


def decode_base64_image(base64_string):
    """
    Decodes a base64 string into PIL Image format
    :param base64_string:
    :type base64_string:
    :return:
    :rtype:
    """
    if base64_string.startswith("data:image"):
        base64_string = base64_string.split(",")[-1]

    image_bytes = base64.b64decode(base64_string)
    image_buffer = io.BytesIO(image_bytes)
    pil_image = Image.open(image_buffer)
    return pil_image


def process_image(image_string):
    """
    Preprocesses the image to be in the format required by the model
    :param image_string:
    :type image_string:
    :return:
    :rtype:
    """
    image = decode_base64_image(image_string)
    img = image.resize((256, 256))

    left = 32 // 2  # 256 - 224
    top = 32 // 2
    right = 32 // 2
    bottom = 32 // 2
    img = img.crop((left, top, img.width - right, img.height - bottom))
    img = np.array(img).transpose((2, 0, 1)) / 256

    means = np.array([0.485, 0.456, 0.406]).reshape((3, 1, 1))
    stds = np.array([0.229, 0.224, 0.225]).reshape((3, 1, 1))

    img = img - means
    img = img / stds

    img_tensor = torch.Tensor(img)

    return img_tensor


def predict(image_string, topk=5, train_on_gpu=False, n_classes=20):
    model = load_model(n_classes=n_classes, model_on_gpu=train_on_gpu, multi_gpu=False)
    # Convert to pytorch tensor
    img_tensor = process_image(image_string)
    if n_classes == 20:
        idx_to_class = {
            0: "Idli",
            1: "burger",
            2: "butter_naan",
            3: "chai",
            4: "chapati",
            5: "chole_bhature",
            6: "dal_makhani",
            7: "dhokla",
            8: "fried_rice",
            9: "jalebi",
            10: "kaathi_rolls",
            11: "kadai_paneer",
            12: "kulfi",
            13: "masala_dosa",
            14: "momos",
            15: "paani_puri",
            16: "pakode",
            17: "pav_bhaji",
            18: "pizza",
            19: "samosa",
        }
    elif n_classes == 14:
        idx_to_class = {
            0: "AlooParatha",
            1: "Biryani",
            2: "ButterChicken",
            3: "CholeBhature",
            4: "DalMakhani",
            5: "Dosa",
            6: "GulabJamuninbowl",
            7: "Idli",
            8: "Jalebionplate",
            9: "Naan",
            10: "PaneerTikka",
            11: "PaniPuri",
            12: "Samosas",
            13: "VadaPav",
        }
    else:
        raise ValueError("Invalid number of classes")
    # Resize
    if train_on_gpu:
        img_tensor = img_tensor.view(1, 3, 224, 224).cuda()
    else:
        img_tensor = img_tensor.view(1, 3, 224, 224)

    # Set to evaluation
    with torch.no_grad():
        model.eval()
        out = model(img_tensor)
        ps = torch.exp(out)

        topk, topclass = ps.topk(topk, dim=1)

        top_classes = [idx_to_class[class_] for class_ in topclass.cpu().numpy()[0]]
        top_p = topk.cpu().numpy()[0]

        return top_p, top_classes


def load_model(n_classes=20, model_on_gpu=False, multi_gpu=False):
    model = models.resnet50(pretrained=True)
    # Make sure to set parameters as not trainable
    for param in model.parameters():
        param.requires_grad = False
    n_inputs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(n_inputs, 256),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(256, n_classes),
        nn.LogSoftmax(dim=1),
    )
    if n_classes == 14:
        model_path = "model/resnet50-pretrain_old.pt"
    elif n_classes == 20:
        model_path = "model/resnet50-pretrain_new.pt"
    else:
        raise ValueError("Invalid number of classes")
    if model_on_gpu:
        model_dict = torch.load(model_path)
    else:
        model_dict = torch.load(model_path, map_location="cpu")

    if multi_gpu:
        model = nn.DataParallel(model_dict.load_state_dict(model))
    else:
        from collections import OrderedDict

        new_state_dict = OrderedDict()
        for k, v in model_dict.items():
            name = k[7:]
            new_state_dict[name] = v

        model.load_state_dict(new_state_dict)
    return model