import cv2
import numpy as np
import torch
from torchvision.transforms import transforms

from models.hrnet import HRNet
from models.detectors.YOLOv3 import YOLOv3


class THRNet:
    """
    Clase HRNet.

    La clase proporciona un método simple y personalizable para cargar la red HRNet, cargar el oficial pre-entrenado
    pesos y predecir la pose humana en imágenes individuales.
    """

    def __init__(self,
                 c,
                 nof_joints,
                 checkpoint_path,
                 resolution=(384, 288),
                 interpolation=cv2.INTER_CUBIC,   
                 multiperson=True,   
                 yolo_model_def="./models/detectors/yolo/config/yolov3.cfg",
                 yolo_class_path="./models/detectors/yolo/data/coco.names",
                 yolo_weights_path="./models/detectors/yolo/weights/yolov3.weights",
                 device=torch.device("cpu")):
        

        self.c = c
        self.nof_joints = nof_joints
        self.checkpoint_path = checkpoint_path
        self.resolution = resolution  #en la forma (alto, ancho) como en la implementación original
        self.interpolation = interpolation
        self.multiperson = multiperson
        self.max_batch_size = max_batch_size
        self.yolo_model_def = yolo_model_def
        self.yolo_class_path = yolo_class_path
        self.yolo_weights_path = yolo_weights_path
        self.device = device

        self.model = HRNet(c=c, nof_joints=nof_joints).to(device)
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        if 'model' in checkpoint:
            self.model.load_state_dict(checkpoint['model'])
        else:
            self.model.load_state_dict(checkpoint)
        self.model.eval()

        if not self.multiperson:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

        else:
            self.detector = YOLOv3(model_def=yolo_model_def,
                                   class_path=yolo_class_path,
                                   weights_path=yolo_weights_path,
                                   classes=('person',),
                                   max_batch_size=self.max_batch_size,
                                   device=device)
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((self.resolution[0], self.resolution[1])),  # (height, width)
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

        pass

    def predict(self, image):
        """
        Predice la pose humana en una sola imagen.
        """
        if len(image.shape) == 3:
            return self._predict_single(image)
        elif len(image.shape) == 4:
            return self._predict_batch(image)
        else:
            raise ValueError('Mal formato de imagen.')

    def _predict_single(self, image):
        if not self.multiperson:
            old_res = image.shape
            if self.resolution is not None:
                image = cv2.resize(
                    image,
                    (self.resolution[1], self.resolution[0]),  # (ancho, alto)
                    interpolation=self.interpolation
                )

            images = self.transform(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)).unsqueeze(dim=0)
            boxes = np.asarray([[0, 0, old_res[1], old_res[0]]], dtype=np.float32)  # [x1, y1, x2, y2]

        else:
            detections = self.detector.predict_single(image)

            boxes = []
            if detections is not None:
                images = torch.empty((len(detections), 3, self.resolution[0], self.resolution[1]))  
                for i, (x1, y1, x2, y2, conf, cls_conf, cls_pred) in enumerate(detections):
                    x1 = int(round(x1.item()))
                    x2 = int(round(x2.item()))
                    y1 = int(round(y1.item()))
                    y2 = int(round(y2.item()))

                    # Adapte las detecciones para que coincidan con la relación de aspecto de entrada de HRNet 
                    correction_factor = self.resolution[0] / self.resolution[1] * (x2 - x1) / (y2 - y1)
                    if correction_factor > 1:
                        #  incrementando
                        center = y1 + (y2 - y1) // 2
                        length = int(round((y2 - y1) * correction_factor))
                        y1 = max(0, center - length // 2)
                        y2 = min(image.shape[0], center + length // 2)
                    elif correction_factor < 1:
                        # seguimos incrementando
                        center = x1 + (x2 - x1) // 2
                        length = int(round((x2 - x1) * 1 / correction_factor))
                        x1 = max(0, center - length // 2)
                        x2 = min(image.shape[1], center + length // 2)

                    boxes.append([x1, y1, x2, y2])
                    images[i] = self.transform(image[y1:y2, x1:x2, ::-1])

            else:
                images = torch.empty((0, 3, self.resolution[0], self.resolution[1]))  # (height, width)

            boxes = np.asarray(boxes, dtype=np.int32)

        if images.shape[0] > 0:
            images = images.to(self.device)

            with torch.no_grad():
                if len(images) <= self.max_batch_size:
                    out = self.model(images)

                else:
                    out = torch.empty(
                        (images.shape[0], self.nof_joints, self.resolution[0] // 4, self.resolution[1] // 4)
                    ).to(self.device)
                    for i in range(0, len(images), self.max_batch_size):
                        out[i:i + self.max_batch_size] = self.model(images[i:i + self.max_batch_size])

            out = out.detach().cpu().numpy()
            pts = np.empty((out.shape[0], out.shape[1], 3), dtype=np.float32)
            # Para cada humano, para cada articulación: x, y, confianza
            for i, human in enumerate(out):
                for j, joint in enumerate(human):
                    pt = np.unravel_index(np.argmax(joint), (self.resolution[0] // 4, self.resolution[1] // 4))
                    pts[i, j, 0] = pt[0] * 1. / (self.resolution[0] // 4) * (boxes[i][3] - boxes[i][1]) + boxes[i][1]
                    pts[i, j, 1] = pt[1] * 1. / (self.resolution[1] // 4) * (boxes[i][2] - boxes[i][0]) + boxes[i][0]
                    pts[i, j, 2] = joint[pt]

        else:
            pts = np.empty((0, 0, 3), dtype=np.float32)

        return pts

    def _predict_batch(self, images):
        if not self.multiperson:
            old_res = images[0].shape

            if self.resolution is not None:
                images_tensor = torch.empty(images.shape[0], 3, self.resolution[0], self.resolution[1])
            else:
                images_tensor = torch.empty(images.shape[0], 3, images.shape[1], images.shape[2])

            for i, image in enumerate(images):
                if self.resolution is not None:
                    image = cv2.resize(
                        image,
                        (self.resolution[1], self.resolution[0]),
                        interpolation=self.interpolation
                    )

                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                images_tensor[i] = self.transform(image)

            images = images_tensor
            boxes = np.repeat(
                np.asarray([[0, 0, old_res[1], old_res[0]]], dtype=np.float32), len(images), axis=0
            )  # [x1, y1, x2, y2]

        else:
            image_detections = self.detector.predict(images)

            boxes = []
            images_tensor = []
            for d, detections in enumerate(image_detections):
                image = images[d]
                boxes_image = []
                if detections is not None:
                    images_tensor_image = torch.empty(
                        (len(detections), 3, self.resolution[0], self.resolution[1]))  # (height, width)
                    for i, (x1, y1, x2, y2, conf, cls_conf, cls_pred) in enumerate(detections):
                        x1 = int(round(x1.item()))
                        x2 = int(round(x2.item()))
                        y1 = int(round(y1.item()))
                        y2 = int(round(y2.item()))

                    
                        correction_factor = self.resolution[0] / self.resolution[1] * (x2 - x1) / (y2 - y1)
                        if correction_factor > 1:
                           
                            center = y1 + (y2 - y1) // 2
                            length = int(round((y2 - y1) * correction_factor))
                            y1 = max(0, center - length // 2)
                            y2 = min(image.shape[0], center + length // 2)
                        elif correction_factor < 1:
                           
                            center = x1 + (x2 - x1) // 2
                            length = int(round((x2 - x1) * 1 / correction_factor))
                            x1 = max(0, center - length // 2)
                            x2 = min(image.shape[1], center + length // 2)

                        boxes_image.append([x1, y1, x2, y2])
                        images_tensor_image[i] = self.transform(image[y1:y2, x1:x2, ::-1])

                else:
                    images_tensor_image = torch.empty((0, 3, self.resolution[0], self.resolution[1]))  # (height, width)

                # apilar todas las imágenes y cuadros en listas individuales
                images_tensor.extend(images_tensor_image)
                boxes.extend(boxes_image)

            # convertir listas en tensores / np.ndarrays
            images = torch.tensor(np.stack(images_tensor))
            boxes = np.asarray(boxes, dtype=np.int32)

        images = images.to(self.device)

        with torch.no_grad():
            if len(images) <= self.max_batch_size:
                out = self.model(images)

            else:
                out = torch.empty(
                    (images.shape[0], self.nof_joints, self.resolution[0] // 4, self.resolution[1] // 4)
                ).to(self.device)
                for i in range(0, len(images), self.max_batch_size):
                    out[i:i + self.max_batch_size] = self.model(images[i:i + self.max_batch_size])

        out = out.detach().cpu().numpy()
        pts = np.empty((out.shape[0], out.shape[1], 3), dtype=np.float32)
       
        for i, human in enumerate(out):
            for j, joint in enumerate(human):
                pt = np.unravel_index(np.argmax(joint), (self.resolution[0] // 4, self.resolution[1] // 4))
          
                pts[i, j, 0] = pt[0] * 1. / (self.resolution[0] // 4) * (boxes[i][3] - boxes[i][1]) + boxes[i][1]
                pts[i, j, 1] = pt[1] * 1. / (self.resolution[1] // 4) * (boxes[i][2] - boxes[i][0]) + boxes[i][0]
                pts[i, j, 2] = joint[pt]

        if self.multiperson:
            # volver a agregar el eje de lote eliminado (n)
            pts_batch = []
            index = 0
            for detections in image_detections:
                if detections is not None:
                    pts_batch.append(pts[index:index + len(detections)])
                    index += len(detections)
                else:
                    pts_batch.append(np.zeros((0, self.nof_joints, 3), dtype=np.float32))
            pts = pts_batch

        else:
            pts = np.expand_dims(pts, axis=1)

        return pts
