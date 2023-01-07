import os
import numpy as np
import glob
import yaml
from math import ceil
import ntpath
from argparse import ArgumentParser
import json

class SplitDataSet():
    def __init__(self, args):
        self.yaml_path = '/workspace/customized/hyperparameters/parameters.yaml'
        self.dst_dataset_path = '/workspace/customized/dataset'
        self.args = args
        self._load_model_type()
        # self.split_dataset_json_path = os.path.join('/workspace', 'result', 'split_dataset.json')
        # self.split_dataset_custom_json_path = os.path.join('/workspace', 'result', 'split_dataset_custom.json')

    def _load_model_type(self):
        with open(self.yaml_path, 'r') as file:
            hpr_list = yaml.load(file, Loader=yaml.FullLoader)
            self.model_type = hpr_list['model_type']

            assert self.model_type in ('classification', 'object_detection', 'instance_segmentation', 'semantic_segmentation')

        if self.model_type == 'object_detection':
            self.label_extension = '.xml'
        elif self.model_type in ('instance_segmentation', 'semantic_segmentation'):
            self.label_extension = '.json'
        elif self.model_type == 'classification':
            self.label_extension = '.txt'

    def run(self):
        if self.args.reset_json:
            self.reset_split_dataset_json(os.path.join('/workspace', 'aidms', 'results', 'split_dataset.json'))
            self.reset_split_dataset_json(os.path.join('/workspace', 'aidms', 'results', 'split_dataset_custom.json'))
        else:
            # clear all previous symbolic links of dataset:
            self._delete_pre_symbolic_link()
            
            self.create_symbolic_link(
                os.path.join('/workspace', 'aidms', 'results', 'split_dataset.json'),
                os.path.join('/workspace', 'aidms', 'dataset', 'all')
                )
            self.create_symbolic_link(
                os.path.join('/workspace', 'aidms', 'results', 'split_dataset_custom.json'),
                os.path.join('/workspace', 'aidms', 'results', 'upload_dataset')
                )

    def create_symbolic_link(self, split_dataset_json_path, src_dataset_path):
        
        labels_path = glob.glob(os.path.join(src_dataset_path, '*' + self.label_extension))
        lables_num = len(labels_path)
        labels_idx = np.arange(lables_num)
        if self.args.customize_split:
            with open(split_dataset_json_path) as f:
                split_dataset_json = json.load(f)
            train_idx = list(map(lambda f: labels_path.index(os.path.join(src_dataset_path, os.path.splitext(f)[0] + self.label_extension)), split_dataset_json['train']))
            val_idx = list(map(lambda f: labels_path.index(os.path.join(src_dataset_path, os.path.splitext(f)[0] + self.label_extension)), split_dataset_json['validation']))
            test_idx = list(map(lambda f: labels_path.index(os.path.join(src_dataset_path, os.path.splitext(f)[0] + self.label_extension)), split_dataset_json['test']))
        else:
            # shuffle idx
            np.random.seed(self.args.random_seed)
            np.random.shuffle(labels_idx)

            train_labels_num = int((self.args.train_ratio)*lables_num)
            train_val_labels_num = int((self.args.train_ratio+self.args.val_ratio)*lables_num)
            all_labels_num = int((self.args.train_ratio + self.args.val_ratio + self.args.test_ratio)*lables_num)
            # split train, validation and test part index
            train_idx, val_idx, test_idx, _ = np.split(labels_idx, [train_labels_num, train_val_labels_num, all_labels_num])
        labels_path = np.asarray(labels_path)
        # train/val/test are same dataset:
        # train_idx = val_idx = test_idx =  np.arange(lables_num)

        # assign training
        all_imgs_basename, train_imgs_basename, validation_imgs_basename, test_imgs_basename = [], [], [], []
        for label_path in labels_path[train_idx]:
            self._create_symbolic_by_set(label_path, 'train')
            train_imgs_basename.append(self._load_img_basename_by_labelfile(label_path))
        # assign validation
        for label_path in labels_path[val_idx]:
            self._create_symbolic_by_set(label_path, 'validation')
            validation_imgs_basename.append(self._load_img_basename_by_labelfile(label_path))
        # assign test
        for label_path in labels_path[test_idx]:
            self._create_symbolic_by_set(label_path, 'test')
            test_imgs_basename.append(self._load_img_basename_by_labelfile(label_path))
        
        for label_path in labels_path:
            all_imgs_basename.append(self._load_img_basename_by_labelfile(label_path))
        
        self._write_split_dataet_json(
            {'all': all_imgs_basename, 'train': train_imgs_basename, 'validation': validation_imgs_basename, 'test': test_imgs_basename},
            split_dataset_json_path
        )
    # delete previous soft links :
    def _delete_pre_symbolic_link(self):
        os.system(f'rm /workspace/customized/dataset/train/*')
        os.system(f'rm /workspace/customized/dataset/validation/*')
        os.system(f'rm /workspace/customized/dataset/test/*')

    def _create_symbolic_by_set(self, scr_path, dst_folder):

        # define destination of symbolic link of label file
        dst_path = os.path.join(self.dst_dataset_path, dst_folder, os.path.basename(scr_path))
        assert not os.path.exists(dst_path), f'{dst_path} exists!'
        # load label file's corresponding image basename
        img_basename = self._load_img_basename_by_labelfile(scr_path)
        # define source of image path
        scr_folder = os.path.dirname(scr_path)
        img_src_path = os.path.join(scr_folder, img_basename)
        # define destination of symbolic link of image file 
        img_dst_path = os.path.join(self.dst_dataset_path, dst_folder, img_basename)
        
        # create symbolic link of image file
        os.symlink(img_src_path, img_dst_path)
        # create symbolic link of label file
        os.symlink(scr_path, dst_path)

    def _write_split_dataet_json(self, split_dataset_json, split_dataset_json_path):
        with open(split_dataset_json_path, 'w') as f:
            f.write(json.dumps(split_dataset_json))

    def reset_split_dataset_json(self, split_dataset_json_path):

        self._delete_pre_symbolic_link()
        os.system(f'rm /workspace/aidms/results/upload_dataset/*')

        dataset = ['train', 'validation', 'test']
        if os.path.basename(split_dataset_json_path) == 'split_dataset_custom.json':
            dataset.append('all') 

        with open(split_dataset_json_path, 'r') as f:
            split_dataset_json = json.loads(f.read())
        for dataset in dataset:
            split_dataset_json[dataset] = []
        self._write_split_dataet_json(split_dataset_json, split_dataset_json_path)

    def _load_img_basename_by_labelfile(self, label_path):
        
        if self.model_type == 'object_detection':
            import xml.etree.ElementTree as ET
            tree = ET.parse(label_path)
            root = tree.getroot()
            try:
                img_name = root.find('path').text
            except Exception as e:
                img_name = root.find('filename').text
            img_basename = ntpath.basename(img_name)

        elif self.model_type in ('instance_segmentation', 'semantic_segmentation'):
            import json 
            with open(label_path, 'r') as  f:
                img_json = json.load(f)
            img_basename = img_json['imagePath']
            img_basename = ntpath.basename(img_basename)
        
        elif self.model_type == 'classification':
            img_extensions = ('jpg', 'JPG', 'jpeg', 'JPEG', 'gif', 'GIF', 'png', 'PNG', 'bmp', 'BMP')
            label_dirname = os.path.dirname(label_path)
            for img_extension in img_extensions:
                test_img_basename = os.path.splitext(os.path.basename(label_path))[0]+ f'.{img_extension}'
                img_path = os.path.join(label_dirname, test_img_basename)
                if os.path.isfile(img_path):
                    img_basename = test_img_basename
                    break
                
        return img_basename

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--train_ratio", type=float, default=0.8, help="train_ratio")
    parser.add_argument("--val_ratio", type=float, default=0.1, help="val_ratio")
    parser.add_argument("--test_ratio", type=float, default=0.1, help="test_ratio")
    parser.add_argument("--random_seed", type=int, default=1, help="random_seed")
    parser.add_argument("--customize_split", type=bool, default=False, help="customize_split")
    parser.add_argument("--reset_json", type=bool, default=False, help="reset_json")
    args = parser.parse_args()
    sd = SplitDataSet(args)
    sd.run()
        
    
    