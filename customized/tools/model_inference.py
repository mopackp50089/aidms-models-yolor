import numpy as np
import pickle
import sys
sys.path.append('/workspace/aidms/lib/inference_class')
from objectdetection import ObjectDetection
import os
import cv2

#harry
sys.path.append('/workspace/customized/models/yolor/')
from utils import datasets, general
from utils.torch_utils import select_device
from models.models import Darknet
sys.path.append('/workspace/customized/models/')
from model_parameters import Model_Parameters
import torch
device = select_device()



class Model(ObjectDetection):
    def __init__(self, mode):
        '''
        Output:
            self.model_weight_name:
                (The model weight name for loading)
                    example (YOLOv4):
                        self.model_weight_name = 'yolov4-leadtek_output_final.weights'
        '''
        # self.model_weight_name = 'Add your model weight name'
        self.model_weight_name = ''

        # Get init from /workspace/customized/models/model_parameters.py
        model_parameters = Model_Parameters(step='training')
        self.yolor_cfg = model_parameters.save_backbone_cfg
        self.save_weight_path = model_parameters.save_weight_path

        super(Model, self).__init__(mode)
        self.conf_thres = self.parameters['inference', 'advanced_para', 'CONF_THRESHOLD'][0]
        self.iou_thres = self.parameters['inference', 'advanced_para', 'IOU_THRESHOLD'][0]


    def _load_model(self):
        '''
        Input:
            self.model_weight_path
                (model weight name with absolute path, please use this path to load model weight)
        Output:
            self.model:
                (Instance of Class of custimized model)
                    example (YOLOv4):
                        network, class_names, class_colors = darknet.load_network(
                            config_file=self.config_file,
                            data_file=self.data_file,
                            # weights=f"/workspace/result/{self.select_result_id}/model_weight/yolov4-leadtek_output_last.weights",
                            weights=self.model_weight_path, 
                            batch_size=1)
        '''
        #load model weight
        ckpt = torch.load(self.save_weight_path, map_location=device)  
        model = Darknet(cfg=self.yolor_cfg, img_size=self.img_size_w).cuda() #yolor-Darknet
        state_dict = {k: v for k, v in ckpt['model'].items() if model.state_dict()[k].numel() == v.numel()}
        model.load_state_dict(state_dict, strict=False)
        model.to(device).eval()
        #set your model to self.model
        self.model = model
    
    def process_infer_images(self, images):
        '''
        (
            B = batch size
            W = image width
            H = image height
            C = number of image channel
        )
        Input: 
        1. images:
                numpy array: shape:(B, W, H, C)
        Output:
        1. your model inference data:
                dictionary : model_inference_data
        '''
        #step1
        #Use image to construct the data format required by your model
        img0 = images
        image_init_shape = img0[0].shape
        for one_batch in range(len(img0)):
            #yolor/utils/dataset.py
            img0[one_batch] = datasets.letterbox(img0[one_batch], new_shape=self.img_size_w, auto_size=64)[0]
            # Convert
            img0[one_batch] = np.ascontiguousarray(img0[one_batch])

        # Convert (B, W, H, C) -> (B, C, W, H)
        img = img0[:, :, :, ::-1].transpose(0, 3, 1, 2).copy()  # BGR to RGB, to 3x416x416
        
        #step2 
        #Pack the required data into a dictionary
        model_inference_data = {}

        model_inference_data['img'] = img
        model_inference_data['img0'] = img0
        model_inference_data['image_init_shape'] = image_init_shape
        
        return model_inference_data
        
    def _infer_model(self, images):
        '''
        (
            B = batch size
            N = number of objects in one image 
            4 = [x0(left), y0(bottom), x1(right), y1(top)]
        )

        Input:
        1. dictionary : model_inference_data 

        Output: 
        1.  self.scores : 
                numpy array: shape:(B, N); example:[[0.8, 0.7, 0.9], [0.4, 0.7], ...]
        2.  self.classes : 
                numpy array: shape:(B, N); example:[[1, 2, 1], [0, 4], ...]
        3.  self.bboxes : [left, bottom, right, top]
                numpy array: shape:(B, N, 4); example:
                [
                    [[1,2,4,6], [4,5,7,8], [9,10,11,12]], [10, 5, 12, 10], [1, 2, 4, 3]]
                ]                             
        4.  self.masks : 
                numpy array: shape:(B, N, W, H); The element in W*H is 0/1 (binary)
        '''
        model_inference_data = self.process_infer_images(images)
        img = model_inference_data['img']
        img0 = model_inference_data['img0']
        image_init_shape = model_inference_data['image_init_shape']
        
        half = device.type != 'cpu'
        #faster
        if half:
            self.model.half()  # to FP16

        img0 = img0[0] #need 3-diminsion

        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0

        with torch.no_grad():
            out = self.model(img)[0]
            out = general.non_max_suppression(out, self.conf_thres, self.iou_thres)

        B_bboxes, B_scores, B_classes = [], [], []
        for _, det in enumerate(out):     
            #gn = torch.tensor(img0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            #if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
            det[:, :4] = general.scale_coords(img.shape[2:], det[:, :4], image_init_shape).round()
            N_bbox, N_scores, N_classes = [], [], []
            #xy1=top-left, xy2=bottom-right
            for *xyxy, conf, cls in det:
                class_idx = int(cls)
                #bbox = [left, top, right, bottom]
                bbox = [int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])]

            
                N_bbox.append(bbox)
                N_scores.append(float('{:.5f}'.format(conf)))
                N_classes.append(class_idx) # classes = class_idx
            
            B_bboxes.append(N_bbox)
            B_scores.append(N_scores)       
            B_classes.append(N_classes) 
            

        self.bboxes =  np.asarray(B_bboxes)
        self.scores = np.asarray(B_scores)
        self.classes = np.asarray(B_classes)
        #self.bboxes.tolist()[img_idx][obj_idx] 迴圈使用

    def load_class_names(self):
        '''
        Output:
            self.class_map : 
                numpy array: shape:(1, N), N = number of classes; example:['class name 1', 'class name 2', ... 'class name N']
        '''
        with open(f'/workspace/customized/results/weights/class_names.pickle', 'rb') as f: 
            class_name_list = pickle.load(f)
        self.class_map = np.asarray(class_name_list)