import os
import io
import cv2
import json
import base64
import torch
import numpy as np
from PIL import Image
import torch.nn.functional as F
from ts.torch_handler.base_handler import BaseHandler

class ModelHandler(BaseHandler):
    def __init__(self):
        self.input_shape = [512, 512]
        self.orininal_shape = [1024, 1024]
        self.num_classes = 2

        self._context = None
        self.initialized = False
        self.model = None
        self.device = None

    def initialize(self, context):
        #  load the model
        self.manifest = context.manifest
        properties = context.system_properties
        model_dir = properties.get("model_dir")
        # self.device = torch.device("cuda:" + str(properties.get("gpu_id")) if torch.cuda.is_available() else "cpu")

        use_cuda = torch.cuda.is_available() and torch.cuda.device_count() > 0
        self.map_location = 'cuda' if use_cuda else 'cpu'
        self.device = torch.device(self.map_location + ':' +
                                   str(properties.get('gpu_id')
                                       ) if use_cuda else 'cpu')

        # Read model serialize/pt file
        serialized_file = self.manifest['model']['serializedFile']
        model_pt_path = os.path.join(model_dir, serialized_file)
        if not os.path.isfile(model_pt_path):
            raise RuntimeError("Missing the model.pt file")
        self.model = torch.jit.load(model_pt_path, map_location=self.device)
        self.initialized = True

    def cvtColor(self, image):
        if len(np.shape(image)) == 3 and np.shape(image)[2] == 3:
            return image
        else:
            image = image.convert('RGB')
            return image

    def preprocess_input(self, image):
        image /= 255.0
        return image

    def resize_image(self, image, size):
        if isinstance(image, Image.Image):
            iw, ih = image.size
        else:
            iw, ih = image.shape[0:2]
            image = Image.fromarray(image)
        w, h = size

        scale = min(w / iw, h / ih)
        nw = int(iw * scale)
        nh = int(ih * scale)

        image = image.resize((nw, nh), Image.BICUBIC)
        new_image = Image.new('RGB', size, (128, 128, 128))
        new_image.paste(image, ((w - nw) // 2, (h - nh) // 2))

        return new_image, nw, nh

    def  json2pythondic(self, json_data):
        # 使用json.loads()将其解析为Python字典
        data = json.loads(json_data)
        image_base64 = data["image"]
        image_bytes = base64.b64decode(image_base64)
        image_np = np.frombuffer(image_bytes, dtype=np.uint8)
        image_np = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
        return image_np

    def preprocess(self, data):
        # image = data[0].get("data")
        # if image is None:
        #     image = data[0].get("body")
        # image = Image.open(io.BytesIO(image))

        json_data =  data[0]["body"]
        image  = self.json2pythondic(json_data)

        image = self.cvtColor(image)
       #   给图像增加灰条，实现不失真的resize
        image_data, nw, nh = self.resize_image(image, (self.input_shape[1], self.input_shape[0]))
        #   添加上batch_size维度
        image_data = np.expand_dims(np.transpose(self.preprocess_input(np.array(image_data, np.float32)), (2, 0, 1)), 0)
        image_data = torch.from_numpy(image_data)
        return image_data

    def inference(self, image):
        preds = self.model(image)[0]
        preds = F.softmax(preds.permute(1, 2, 0), dim=-1).cpu().detach().numpy()
        preds = cv2.resize(preds, (self.orininal_shape[0], self.orininal_shape[1]), interpolation=cv2.INTER_LINEAR)
        preds = preds.argmax(axis=-1).astype(np.uint8)
        return preds

    def postprocess(self, preds):
        #获取阔论坐标
        image, contours, hierarchy = cv2.findContours(preds, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        loc_arr2 = []
        # 打印每个轮廓的坐标
        for contour in contours:
            loc_arr1 = []
            for point in contour:
                x, y = point[0]
                loc_arr1.append((int(x), int(y)))
            loc_arr2.append(loc_arr1)
        return [loc_arr2]

    def handle(self, data, context):
        data = self.preprocess(data)
        data = self.inference(data)
        data = self.postprocess(data)
        return data