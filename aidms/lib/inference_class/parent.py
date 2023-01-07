import base64
import json
import numpy as np
import cv2
import abc
import yaml
import glob
import os
import math
import pickle
import subprocess
import time
import io
from autolabel_template.autolabel import NpEncoder
import sys
sys.path.append('/workspace/aidms/lib/')
from parameters_class.parameters import Parameters

# create general class for three model-type when doing inference:
class CVParent():
    def __init__(self, mode='inference'):
        self.mode = mode
        self.test_path = self._load_test_images_path()
        self.image_extension = ['*.png', '*.jpeg', '*.jpg', '*.bmp', '*.gif']
        self.image_extension.extend([img_extension.upper() for img_extension in self.image_extension.copy()])
        self.parameters = self._load_hyperparamters()
        # self.img_size = int(self.parameters.training.advanced_para.image_size)
        # self.batch_size = int(self.parameters.inference.default_para.batch_size)
        self.img_size_w = self.parameters['training', 'advanced_para', 'image_size', 'Width'][0]
        self.img_size_h = self.parameters['training', 'advanced_para', 'image_size', 'Height'][0]
        # self.batch_size, _ = self.parameters['training', 'default_para', 'batch_size']
        self._assign_class_color()
        self._load_model()
        self.scores, self.classes, self.bboxes, self.masks  = np.array([]), np.array([]), np.array([]), np.array([])
        self.infer_batch_size = 1

    @abc.abstractmethod
    def _load_model(self):
        '''
        call this function to load model and save model in self.model .

        Example:
        tf.keras.backend.clear_session()
        model = tf.saved_model.load(self.model_weight_path)
        self.model = model
        '''

    @abc.abstractmethod
    def _infer_model(self, images):
        '''
        infer image and save figure(visualized), score and class to:
        input:
            images             (type: numpy array, (batch=1, Width, Height, Channel))
        output:
            self.socres             (type: list, [score1, score2, ...scoreN])
            self.classes            (type: list, [class_name1, class_name2, ...class_nameN])
            self.bboxes             (type: list, [[x0, y0, x1, y1], [x0, y0, x1, y1], ...boxN])
            self.masks              ((type: list, [[w,h], [w,h], ...mask_N]), element of [w,h] is 1 or 0)

        Example:
        self.model.run_on_image(images)
        '''
    @abc.abstractmethod
    def load_class_names(self):
        '''
        load your class_map file, such as: 
        class_map = ['car', 'dog', 'zebra', ...]
        set:
            self.class_map = ['car', 'dog', 'zebra', ...]
        '''    

    def get_model_result_for_client(self, output_image=True, pipeline=False, output_result=True):
        b64_vis_images, ROIs = [], []
        # self.all_classes, self.all_bboxes, self.all_scores, self.all_masks = np.array([]), np.array([]), np.array([]), np.array([])
        self._init_model_results()
        img_batches_generator = self._get_img_arr_batches(get_img_name=False)

        for self.recent_imgs in img_batches_generator:

            self._infer_model(self.recent_imgs)

            self.all_classes.extend(self.class_map[self.classes.astype(int)].tolist())
            self.all_scores.extend(self.scores.tolist())
            self.all_bboxes.extend(self.bboxes.tolist())
            self.all_masks.extend(self.masks.tolist())

            if output_image:
                b64_vis_images.extend(self._visualize_image(op2b64=True))
            t3 = time.time()

            if pipeline and self.model_type in ('object_detection', 'instance_segmentation'):
                ROIs.extend(self._get_ROIs_from_bbox(op2b64=True))

        if not output_result:
            result = {}
        else:
            result = {
                'scores': self.all_scores, 
                'classes': self.all_classes,
                'bboxes': self.all_bboxes,
                'masks': self.all_masks
                }
        
        new_result = result.copy()
        for key, value in result.items():
            if not len(value):
                new_result.pop(key)
            # else:
            #     new_result[key] = value[0]
        
        new_result['images'] = b64_vis_images
        new_result['ROIs'] = ROIs
        
        return new_result

    def get_model_result_for_result(self, save_json=False, save_img=True):
        print(f'Now, inference images ...')
        
        if self.mode == 'test':
            results = []
            result_json = open('/workspace/aidms/tmp/metric/annotations_res.json', 'w')
            result_json.write('[')  # json load double quote.
            wrote_first_object = False

        FPS = []
        all_imgs_name = []
        self._init_model_results()
        img_batches_generator = self._get_img_arr_batches()
        for test_img_names, self.recent_imgs in img_batches_generator:
            # all_imgs_name.extend(test_img_names)
            t1 = time.time()
            self._infer_model(self.recent_imgs)
            t2 = time.time()
            FPS.append(t2-t1)

            if not self.mode == 'inference':
                save_root_path = os.path.join('/workspace/aidms/results', f'model_{self.select_result_id}')
            else:
                save_root_path = '/workspace/result'

            if save_img:
                self.vis_imgs = self._visualize_image(op2b64=False)

                for img_idx, vis_image in enumerate(self.vis_imgs):
                    img_save_path = os.path.join(save_root_path, test_img_names[img_idx])
                    print('Saving image: ', img_save_path)
                    cv2.imwrite(img_save_path, vis_image)
            
            if self.mode == 'test':
                if self.model_type in ('object_detection', 'instance_segmentation'):
                    for img_idx, test_img_name in enumerate(test_img_names):
                        result = dict()
                        result['image_id'] = os.path.splitext(test_img_name)[0] #get '1' from '1.jpg'
                        # result['image_id'] = 0
                        one_img_classes = self.classes.tolist()[img_idx]
                        for obj_idx in range(len(one_img_classes)):
                            result['category_id'] = one_img_classes[obj_idx]
                            result['bbox'] = self._convert_xyxy2xywh(self.bboxes.tolist()[img_idx][obj_idx])
                            result['score'] = self.scores.tolist()[img_idx][obj_idx]
                            one_result = result.copy()
                            if wrote_first_object:
                                result_json.write(',') 
                            result_json.write(str(one_result).replace("'", '"'))  # json load double quote.
                            wrote_first_object = True
                elif self.model_type in ('classification', ):
                    for img_idx, test_img_name in enumerate(test_img_names):
                        one_img_result = dict()
                        one_img_class = self.classes[img_idx][0]
                        one_img_result[test_img_name] = one_img_class
                        if wrote_first_object:
                                result_json.write(',') 
                        result_json.write(str(one_img_result).replace("'", '"'))
                        wrote_first_object = True

        if self.mode == 'test':
            result_json.write(']')
            result_json.close() 
        # self.imgs_name = all_imgs_name

        # result = {
        #     'scores': self.all_scores, 
        #     'classes': self.all_classes,
        #     'bboxes': self.all_bboxes,
        #     # 'masks': self.all_masks
        #     }
        # if save_json:

        #     json_save_path = os.path.join(save_root_path, 'model_results.json')

        #     with open(json_save_path, 'w') as f:
        #         json.dump(result, f)
            

        if self.mode == 'test':
            print('Writing FPS ...')
            t1 = time.time()
            self._write_FPS(np.asarray(FPS)[1:])
            print('FPS has been written.', round(time.time()-t1, 2), 'seconds')
            print('Writing mAP ...')
            t1 = time.time()
            with open('/workspace/aidms/tmp/class_names.pickle', 'wb') as f:
                class_map = self.class_map.tolist()
                class_map.insert(0, '__ignore__')
                pickle.dump(class_map, f)
            if self.model_type != 'classification':
                self._measure_mAP()
            else:
                self._measure_confusion_matrix()
            print('mAP/confusion matrix has been written.', round(time.time()-t1, 2), 'seconds')
        elif self.mode == 'test_uploadimg':
            print('Create auto label files ...')
            t1 = time.time()
            # self._auto_label()
            print('auto label files have been created.', round(time.time()-t1, 2), 'seconds')
        # return result
        return 

    def _init_model_results(self):
        self.all_classes, self.all_bboxes, self.all_scores, self.all_masks = [], [], [], []

    def _get_img_arr_batches(self, get_img_name=True):
        if self.mode == 'inference':
            load_images_generator = self.load_images_from_client()
        else:
            load_images_generator = self.load_images()
        for imgs_name, imgs_array in load_images_generator:      
            if get_img_name:
                yield imgs_name, imgs_array
            else: 
                yield imgs_array

    def _nparr2b64(self, inf_img):
        '''
        server send numpy array, and transfer to decode(base64_endoce_string)
        '''
        retval, buffer = cv2.imencode('.png', inf_img)
        pic_str = base64.b64encode(buffer)
        pic_str = pic_str.decode()
        return pic_str

    def _nparr2b(self, inf_img):
        '''
        server send numpy array, and transfer to decode(base64_endoce_string)
        '''
        retval, buffer = cv2.imencode('.png', inf_img)
        pic_str = buffer.tobytes()
        return pic_str

    def _b642nparr(self, b64_str):
        '''
        Client send decode(base64_endoce_string), and we transfer back to numpy array
        '''
        nparr = np.fromstring(base64.b64decode(b64_str), np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return img

    def _b2nparr(self, b_str):
        '''
        Client send decode(base64_endoce_string), and we transfer back to numpy array
        '''
        t1 = time.time()
        nparr = np.fromstring(b_str, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        print(time.time()-t1, 'byte to numpy array')
        return img
    
    def _resize(self, img):
        img = cv2.resize(img, (self.img_size_w, self.img_size_h))
        return img

    def _get_crop_img(self, img, bbox):
        x0, y0, x1, y1 = bbox
        w, h = x1 - x0, y1 - y0
        crop_img = img[y0:y0+h, x0:x0+w]

        return crop_img

    def load_images_from_client(self):
        '''
        Input:
            imgs:
                Dictionary of images {'img1':decode(base64_endoce_string)} 
        '''
        assert isinstance(self.imgs_json, dict)#check is json
        print('*'*100, '\n', 'Inference mode', '\n', '*'*100)
        imgs_nparr, imgs_name = [], []
        for img_name in self.imgs_json.keys():
            # b64
            try:
                b64_str = self.imgs_json[img_name]
                img_nparr = self._b642nparr(b64_str)
            # or byte string
            except Exception as e:
                print(e)
                f = self.imgs_json[img_name]
                img_name = f.filename
                image_data = io.BytesIO()
                f.save(image_data)
                img_nparr = self._b2nparr(image_data.getvalue())
            img_nparr = self._resize(img_nparr)
            imgs_name.append(img_name)
            imgs_nparr.append(img_nparr)

            if len(imgs_nparr) == self.infer_batch_size:
                # self.imgs_name = imgs_name
                # self.images = np.asarray(imgs_nparr)
                yield imgs_name, np.asarray(imgs_nparr)
                imgs_nparr, imgs_name = [], []

        # self.images = np.asarray(imgs_nparr)

    def load_images_name_from_client(self, imgs_name_list):
        '''
        Input:
            imgs:
                Dictionary of images {'img1':decode(base64_endoce_string)} 
        '''
        assert isinstance(imgs_name_list, list)#check is json
        print('*'*100, '\n', 'Inference mode', '\n', '*'*100)
        imgs_nparr = []
        for img_name in imgs_name_list:
            img_nparr = cv2.imread(os.path.join(self.test_path, img_name))
            img_nparr = self._resize(img_nparr)
            imgs_nparr.append(img_nparr)
        print('Get ', len(imgs_nparr), 'images')

        self.imgs_name = imgs_name_list
        self.images = np.asarray(imgs_nparr)

    def load_images(self):
        print('*'*100, '\n', 'Test mode', '\n', '*'*100)
        all_files, imgs_nparr, imgs_name = [], [], []
        for ext in self.image_extension:
            all_files.extend(glob.glob(os.path.join(self.test_path, ext)))
        assert len(all_files)!=0, f'The {self.test_path} has no any images, please check it !'
        t1 = time.time()
        for img_file in all_files:
            img_nparr = cv2.imread(img_file)
            img_nparr = self._resize(img_nparr)
            imgs_nparr.append(img_nparr)
            imgs_name.append(os.path.basename(img_file))
            if len(imgs_nparr) == self.infer_batch_size:
                # self.imgs_name = imgs_name
                # self.images = np.asarray(imgs_nparr)
                yield imgs_name, np.asarray(imgs_nparr)
                imgs_nparr, imgs_name = [], []
        # print(f'"Read and resize inference images" Spend time: {round(time.time()-t1, 2)}')

        # self.imgs_name = imgs_name
        # self.images = np.asarray(imgs_nparr)

    # def _load_yaml_path(self, mode):
    #     assert mode in ('test', 'test_uploadimg', 'inference')
    #     if mode == 'inference':
    #         yaml_path = '/workspace/parameters/parameters.yaml'
    #     else:
    #         yaml_path = '/workspace/result/parameters_cluster.yaml'
        
    #     return yaml_path

    def _load_test_images_path(self):
        if self.mode == 'test_uploadimg':
            test_images_path = '/workspace/aidms/results/upload_images'
        elif self.mode == 'test':
            test_images_path = '/workspace/customized/dataset/test'
        else:
            test_images_path = '/workspace/dataset/inference'
        
        return test_images_path

    def _load_hyperparamters(self):
        # if self.mode in ('test', 'test_uploadimg'):
        #     parameters, self.model_type, self.select_result_id = Parameters.get_hyperparamters(step='training')
        # elif self.mode in ('inference'):
        #     parameters, self.model_type, self.select_result_id = Parameters.get_hyperparamters(step='inference')
        if self.mode in ('test', 'test_uploadimg'):
            parameters = Parameters(step='training')
            self.model_type, self.select_result_id = parameters.model_type, str(parameters.select_result_id)
        elif self.mode in ('inference'):
            parameters = Parameters(step='inference')
            self.model_type, self.select_result_id = parameters.model_type, str(parameters.select_result_id)
 
        return parameters

    ###################################
    #visualization start ##############
    ###################################
    def _generate_random_color(self):
        color = tuple(np.random.randint(0, 255, 3).tolist())
        return tuple(color)

    def _assign_class_color(self):
        '''
        input:
            class_map:
                ['apple', 'banana']
        output:
            classes_color:
                {
                    'apple':array(0-255, 0-255, 0-255),
                    'banana':array(0-255, 0-255, 0-255),
                }
        '''
        self.load_class_names()
        classes_color = dict()
        for class_name in self.class_map:
            color = self._generate_random_color()
            classes_color[class_name] = color
        if self.model_type == 'semantic_segmentation':
            classes_color['_background_'] = np.asarray([0, 0, 0])
        self._classes_color = classes_color

    def _apply_class(self, image, class_name, color):
        # font
        font = cv2.FONT_HERSHEY_SIMPLEX
        # fontScale
        fontScale = 1
        # Line thickness of 2 px
        thickness = 2
        print(self.img_size_h, self.img_size_w)
        textSize = cv2.getTextSize(class_name, fontFace=font, fontScale=fontScale, thickness=thickness)
        delta_w, delta_h = textSize[0]
        fix_org = (0, self.img_size_h-delta_h)
        image = cv2.putText(image, class_name, fix_org, font, fontScale, 
                        color, thickness, cv2.LINE_AA, False)

        return image

    def _apply_bbox(self, image, class_name, box, confidence, color):
        # draw parameters:
        font_face = cv2.FONT_HERSHEY_SIMPLEX
        x0, y0, x1, y1 = tuple(box)
        
        # font size decision rules
        img_area = self.img_size_w * self.img_size_h
        box_area = (x1-x0) * (y1-y0)
        area_ratio = box_area/img_area
        font_scale = 2.45 * area_ratio + 0.4
        # if area_ratio >= 0.25:
        #     font_scale = 0.8
        # elif area_ratio < 0.25 and area_ratio > 0.01:
        #     font_scale = 0.5
        # else:
        #     font_scale = 0.2
        # print('font_scale', font_scale)
        # font_scale = 0.8
        font_thickness = 1

        text = f'{class_name} {math.ceil(confidence*100)} %'
        labelSize = cv2.getTextSize(text, font_face, font_scale, font_thickness)

        x1_delt = labelSize[0][0]
        y1_delt = labelSize[0][1]

        bbox_thickness = math.ceil(font_scale)
        # print('bbox_thickness: ', bbox_thickness)

        # bbox:
        cv2.rectangle(image, (x0, y0), (x1, y1), color=color, thickness=bbox_thickness)
        
        # text outside bbox:
        cv2.rectangle(image, (x0, y0-y1_delt-font_thickness), (x0+x1_delt, y0), color=color, thickness=-1)
        cv2.putText(image, text, (x0, y0-font_thickness), font_face, font_scale, (0, 255, 255), font_thickness, cv2.LINE_AA)

        # text inside bbox:
        # cv2.rectangle(image, (x0, y0), (x0+x1_delt, y0+y1_delt), color=color, thickness=-1)
        # cv2.putText(image, text, (x0, y0+y1_delt), font_face, font_scale, (0, 255, 255), font_thickness, cv2.LINE_AA)
        
        return image     

    def _apply_mask(self, image, mask, color, alpha=0.7):
        for c in range(3):
            image[:,:,c] = np.where(mask == 1,
                                image[:, :, c] *
                                (1 - alpha) + alpha * color[c],
                                image[:, :, c])
        return image

    def _apply_semantic_mask(self, image, mask, alpha=0.7):
        '''
        draw mask using semantic mask
        '''
        for class_idx, class_color in enumerate(list(self._classes_color.values())):
            for c in range(3):
                image[:,:,c] = np.where(mask == class_idx,
                                image[:, :, c] *
                                (1 - alpha) + alpha * class_color[c],
                                image[:, :, c])
        
        return image

    def _visualize_image(self, op2b64=True):
        
        vis_imgs = []
        for img_idx, image in enumerate(self.recent_imgs):  # number of images
            if self.model_type in ('classification', 'object_detection', 'instance_segmentation'):
                all_img_pred_classes = self.classes
                one_img_pred_classes = all_img_pred_classes[img_idx]
                if len(one_img_pred_classes) == 0: # If there is no any object in the image, just return original image
                    if op2b64:
                        # image = self._nparr2b64(image)
                        image = self._nparr2b(image)
                    vis_imgs.append(image)
                    continue
                one_img_pred_classes_name = self.class_map[one_img_pred_classes]
                # one_img_pred_classes_name = class_map[one_img_pred_classes] # map '1' to 'apple' ... If self.classes is 1 not 'apple'. Use it!
                for obj_idx in range(len(one_img_pred_classes_name)): # I just randomly choose one_img_pred_classes_name to get number of object of one image
                    one_obj_scores = self.scores[img_idx][obj_idx]
                    one_obj_class_name = one_img_pred_classes_name[obj_idx]
                    if len(self.bboxes):
                        one_obj_bbox = self.bboxes[img_idx][obj_idx].astype('int32') # To int32, because x0y0x1y1 have to be int.
                        image = self._apply_bbox(image, one_obj_class_name, one_obj_bbox, one_obj_scores, self._classes_color[one_obj_class_name])
                    # if just classification:
                    elif not len(self.masks):
                        image = self._apply_class(image, one_obj_class_name, self._classes_color[one_obj_class_name])
                    if len(self.masks):
                        one_obj_mask = self.masks[img_idx][obj_idx]
                        image =  self._apply_mask(image, one_obj_mask, self._generate_random_color(), alpha=0.5)
                        # print(one_obj_mask.shape, self._generate_random_color())
            elif self.model_type in ('semantic_segmentation', ):
                image_mask = self.masks[img_idx]
                image = self._apply_semantic_mask(image, image_mask)
            if op2b64:
                # image = self._nparr2b64(image)
                image = self._nparr2b(image)

            vis_imgs.append(image)
        
        return vis_imgs
    
    def _get_ROIs_from_bbox(self, op2b64=True):
        ROIs = []
        for img_idx, image in enumerate(self.recent_imgs):
            bboxes = self.bboxes[img_idx]
            for obj_idx in range(len(bboxes)):
                one_obj_bbox = bboxes[obj_idx].astype('int32') # To int32, because x0y0x1y1 have to be int.
                crop_image = self._get_crop_img(image, one_obj_bbox)
                crop_image = self._nparr2b64(crop_image)
                ROIs.append(crop_image)
        
        return ROIs
    # ############ FPS ############
    def _write_FPS(self, fps):
        with open(f'/workspace/aidms/results/model_{self.select_result_id}/FPS.txt', 'w') as f:
            f.write(str(1/fps.mean()))
    # ############ mAP ############
    def _measure_mAP(self):
        # create answer annotation.json file from testing data.
        # self._create_model_test_output()
        # transform and create model output to annotation_res.json file
        self._transform_test_data_to_coco()
        # measure mAP
        p, _ = self._bash_command(f'python3 /workspace/aidms/lib/inference_class/evaluate_mAP_cocoapi.py {self.select_result_id}')
        
    def _measure_confusion_matrix(self):
        p, _ = self._bash_command(f'python3 /workspace/aidms/lib/inference_class/calculate_classification_metric.py {self.select_result_id}')

    def _transform_test_data_to_coco(self):
        if self.model_type == 'instance_segmentation':
            os.system(f'python3 /workspace/aidms/lib/inference_class/data_format_converter/labelme2coco.py \
                /workspace/customized/dataset/test/ /workspace/aidms/tmp/metric --labels /workspace/aidms/tmp/class_names.pickle \
                --image_size_w {self.img_size_w} --image_size_h {self.img_size_h}')

        if self.model_type == 'object_detection':
            os.system(f'python3 /workspace/aidms/lib/inference_class/data_format_converter/voc2coco/voc2coco.py --ann_dir /workspace/customized/dataset/test/ \
                --labels /workspace/aidms/tmp/class_names.pickle --output /workspace/aidms/tmp/metric/annotations.json --ext xml\
                --image_size_w {self.img_size_w} --image_size_h {self.img_size_h}')
        

    def _bash_command(self, cmd):
        result = subprocess.Popen(['/bin/bash', '-c', cmd], stdout=subprocess.PIPE)
        text = result.communicate()[0]

        return result.returncode, text.decode("utf-8") 

    def _create_model_test_output(self):
        
        results = []
        for img_idx, img_name in enumerate(self.imgs_name):
            result = dict()
            result['image_id'] = os.path.splitext(img_name)[0] #get '1' from '1.jpg'
            # result['image_id'] = 0
            one_img_classes = self.all_classes[img_idx]
            for obj_idx in range(len(one_img_classes)):
                result['category_id'] = one_img_classes[obj_idx]
                result['bbox'] = self._convert_xyxy2xywh(self.all_bboxes[img_idx][obj_idx])
                result['score'] = self.all_scores[img_idx][obj_idx]
                one_result = result.copy()
                results.append(one_result)

            # print('*'*100)
        with open('/workspace/aidms/tmp/metric/annotations_res.json', 'w') as f:
            f.write(str(results).replace("'", '"'))  # json load double quote.

    def _convert_xyxy2xywh(self, xyxy):
        x = xyxy[0]
        y = xyxy[1]
        w = xyxy[2]-xyxy[0]
        h = xyxy[3]-xyxy[1]

        return [x, y, w , h]

    def _auto_label(self):
        for img_idx, img_name in enumerate(self.imgs_name):
            if self.model_type == 'instance_segmentation':
                from imantics import Polygons, Mask
                self._output_auto_label_json(
                    masks=np.asarray(self.all_masks[img_idx]), 
                    classes=self.class_map[self.all_classes[img_idx]], 
                    image_name=img_name,
                    output_result_id=self.select_result_id
                    )
            elif self.model_type == 'object_detection':
                self._output_auto_label_xml(
                    bboxes=self.all_bboxes[img_idx], 
                    classes=self.class_map[self.all_classes[img_idx]], 
                    image_name=img_name, 
                    output_result_id=self.select_result_id)

    
    def _output_auto_label_json(self, masks, classes, image_name, output_result_id):
        """
        image_name: 
                            image path/name
        """
        
        h, w = masks.shape[1:]
        
        color = np.random.randint(0, 255, 3).tolist()
        color.append(128)
        
        one_json_dict = {
            "version": "4.5.5", "flags": {},"shapes": [], "imageHeight": h,
            "imageWidth": w, "imagePath": os.path.basename(image_name), "imageData": None, 
            "fillColor": [255,0,0,128],"lineColor": [0,255,0,128]
                        }

        for mask_idx in range(masks.shape[0]):
            class_name = classes[mask_idx]
            one_instance_dict = {"label":class_name, 'line_color': None, 
                            'fill_color': None, "shape_type": "polygon", 
                            "flags": {}}
            mask = masks[mask_idx,:,:]

            # We found that maskRCNN probably output empty mask!
            if not np.sum(mask!=False):
                continue
                
            point = self._mask2polygons(mask)
            one_instance_dict["points"] = point
            one_json_dict['shapes'].append(one_instance_dict)

        save_path = os.path.splitext(image_name)[0]+'.json'
        save_path = os.path.join(f'/workspace/aidms/results/model_{output_result_id}', save_path)
        print('saving path:', save_path)
        with open(save_path, 'w') as json_file:
            json.dump(one_json_dict, json_file, cls=NpEncoder)

    
    def _mask2polygons(self, mask):

        polygons_reduce = 1
        polygons = Mask(mask).polygons()[0]
        if len(polygons)/(polygons_reduce)<6:
            polygon_reduce_times = int(len(polygons)/6)
            if polygon_reduce_times<1:
                polygon_reduce_times = 1
        else:
            polygon_reduce_times = polygons_reduce
            
        all_x = polygons[0::polygon_reduce_times*2]
        all_y = polygons[1::polygon_reduce_times*2]
        point = [[x, y]for x, y in zip(all_x, all_y)]

        return point

    # Input is only one image information(bounding box, classes ...etc)
    def _output_auto_label_xml(self, bboxes, classes, image_name, output_result_id):
        with open('/workspace/aidms/lib/inference_class/autolabel_template/xml_base_template.txt', 'r') as f:
            base_template = f.read()
        with open('/workspace/aidms/lib/inference_class/autolabel_template/xml_object_template.txt', 'r') as f:
            object_template = f.read()
        objects = ''
        for object_idx in range(len(bboxes)):
            x0, y0, x1, y1 = bboxes[object_idx]
            class_name = classes[object_idx]
            one_object = object_template.format(class_name=class_name, x0=x0, y0=y0, x1=x1, y1=y1)
            objects += one_object
        output_str = base_template.format(folder=None, filename=image_name, path=image_name, 
                                            width=self.img_size_w, height=self.img_size_h, objects=objects)
        xml_basename = os.path.splitext(image_name)[0]+'xml'
        output_path = os.path.join(f'/workspace/aidms/results/model_{output_result_id}', xml_basename)
        print(output_path)
        with open(output_path, 'w') as f:
            f.write(output_str)
        
    # def _get_original_image_size(self):
    #     # select first image 
        

    # def _transform_coordinate_by_image_size_ratio(self, coordinate_array, src_target_size_ratio):


