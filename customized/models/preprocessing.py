import glob
import os
import subprocess
import yaml
import sys
from model_parameters import Model_Parameters
model_parameters = Model_Parameters(step='training')

# sys.path.append('/workspace/aidms')
# from parameters_class.parameters import Parameters
# parameters = Parameters(step='training')

def bash_command(cmd):
    result = subprocess.Popen(['/bin/bash', '-c', cmd], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    std_output, std_error = result.communicate()
    # print(text)
    return result.returncode, std_output.decode("utf-8"), std_error.decode("utf-8")

def clean_previous_tensorboard_files():
    bash_command(f'rm /workspace/customized/results/tensorboard/events.out.tfevents.*')
    print(f'rm /workspace/customized/results/tendsorboard/events.out.tfevents.*')

def get_train_model_idx():
    return_code, stdout, stderr = bash_command(f"cat /workspace/result/parameters_cluster.yaml | grep select_result_id | awk {{'print $2'}}")
    return stdout.strip()
######################################################
# def get_imgs_path_by_dataset(dataset):
#     img_extensions = ('jpg', 'jpeg', 'gif', 'png', 'bmp')
#     imgs_path = []
#     for img_extension in img_extensions:
#         imgs_path.extend(glob.glob(f'/workspace/dataset/{dataset}/*{img_extension}'))

#     return imgs_path
import labelme2coco
# def convert_json_to_coco():
#     datasets = ['train', 'validation', 'test']
#     for dataset in datasets:
#         labelme_folder = f"/workspace/tmp/yolor_dataset/images/{dataset}"
#         save_json_path = f"/workspace/tmp/yolor_dataset/annotations/instances_{dataset}.json"
#         assert os.path.isdir(labelme_folder),f"FileExistsError: {labelme_folder}"
#         labelme2coco.convert(labelme_folder, save_json_path)
#     print(f"convert to coco finish!")

def convert_json_to_coco2():
  
    labelme_folder = f"/HHD/centermask2/typhoon_data"
    save_json_path = f"/HHD/centermask2/typhoon_data.json"
    assert os.path.isdir(labelme_folder),f"FileExistsError: {labelme_folder}"
    labelme2coco.convert(labelme_folder, save_json_path)
    print(f"convert to coco finish!")

# def convert_coco_to_darknet():
#     datasets = ['train', 'validation', 'test']
#     datasets_class_name_list = []
#     for dataset in datasets:
#         json_file_path = f"/workspace/tmp/yolor_dataset/annotations/instances_{dataset}.json"
#         save_labels_path = f"/workspace/tmp/yolor_dataset/labels/{dataset}/"
#         assert os.path.isfile(json_file_path),f"FileExistsError: {json_file_path}"
#         class_name_list = coco2darknet.convert_annotation(json_file_path,save_labels_path)
#         #get class_name
#         datasets_class_name_list.extend(class_name_list)
#     print(f"convert to darknet finish!")
#     get_yolor_dataset_yaml(datasets_class_name_list=datasets_class_name_list)

# def get_yolor_dataset_yaml(datasets_class_name_list):

#     yolor_dataset_dict = {}
#     yolor_dataset_dict['train'] = 'yolor_dataset/images/train/'
#     yolor_dataset_dict['val'] = 'yolor_dataset/images/validation/'
    
#     unique_datasets_class_name_list = list(set(datasets_class_name_list))
#     print(unique_datasets_class_name_list)
#     yolor_dataset_dict['names'] = unique_datasets_class_name_list#class name
#     yolor_dataset_dict['nc'] = len(unique_datasets_class_name_list)#number class
#     print(yolor_dataset_dict)
    
#     with open(f"{parameters.tmp_path}/yolor_dataset.yaml",'w') as file:
#         yaml.dump(yolor_dataset_dict, file)

######################################################
def establish_model_dataset_structure():
    #step1 establish your dataset structure
    datasets = ['train', 'validation', 'test']
    #create labels folder
    for dataset in datasets:
        try:
            os.makedirs(f"{model_parameters.tmp_path}/yolor_dataset/labels/{dataset}")
        except FileExistsError:
            print(f"{model_parameters.tmp_path}/yolor_dataset/labels/{dataset} is exists!")
    #step2 Set image link path
    symbolic_link_root_path = f"{model_parameters.tmp_path}/yolor_dataset/images"
    try:
        os.symlink(model_parameters.aidms_dataset_path, symbolic_link_root_path)
    except FileExistsError:
        print(f"symbolic_link_root_path: {symbolic_link_root_path} is exists!")

from model_tool.XmlToTxt.objectmapper import ObjectMapper
from model_tool.XmlToTxt.reader import Reader

def get_xml_class_names():
    # xml_dir = f"{model_parameters.aidms_dataset_path}/all"
    xml_dir = f"/workspace/aidms/dataset/all"
    reader = Reader(xml_dir)
    xml_files = reader.get_xml_files()

    object_mapper = ObjectMapper()
    annotations = object_mapper.bind_files(xml_files, xml_dir=xml_dir)
    
    dataset_all_images_class_name_set = set()
    for annotation in annotations: # annotation: One Image of Dataset
        for obj_ann in annotation.objects: #obj_ann: One Object of Image
            dataset_all_images_class_name_set.add(obj_ann.name)
    dataset_all_images_class_name_list = list(dataset_all_images_class_name_set)
    dataset_all_images_class_name_list.sort()
    print("func get_xml_class_names() output -> dataset_all_images_class_name_list:",dataset_all_images_class_name_list)
    return dataset_all_images_class_name_list

def get_yolor_dataset_yaml(class_name_list):
    yolor_dataset_dict = {}
    yolor_dataset_dict['train'] = 'yolor_dataset/images/train/'
    yolor_dataset_dict['val'] = 'yolor_dataset/images/validation/'
    
    yolor_dataset_dict['names'] = class_name_list#class name
    yolor_dataset_dict['nc'] = len(class_name_list)#number class

    #produce model yaml
    with open(f"{model_parameters.save_dataset_yaml}",'w') as file:
        yaml.dump(yolor_dataset_dict, file, Dumper=yaml.Dumper)
    
import pickle
def get_class_name_pickle(class_name_list):
    with open(model_parameters.save_classname_pickle, 'wb') as f: 
        pickle.dump(class_name_list, f)

def get_yolor_backbone_cfg(class_name_list):
    cfg_source_path = model_parameters.pre_backbone_cfg
    cfg_target_path = model_parameters.save_backbone_cfg

    num_classes = len(class_name_list)
    num_filters = (num_classes + 5) * 3
    
    with open(file=cfg_target_path, mode='w') as write_f:
        with open(file=cfg_source_path, mode='r') as read_f:
            write_f.write(read_f.read().format(num_classes = str(num_classes), num_filters = str(num_filters)))
    
def get_pre_weight():
    pre_weight_path = model_parameters.pre_weight
    save_weight_path = model_parameters.save_weight_path
    bash_command(f"cp {pre_weight_path} {save_weight_path}")

def convert_dataset_xml_to_yolo():
    #input: class_names.pickle, *.xml files
    #output: *.txt files
    os.chdir("/workspace/customized/models/model_tool/XmlToTxt")
    class_names_path = model_parameters.save_classname_pickle
    datasets = ['train', 'validation', 'test']
    for dataset in datasets:
        xml_dir = f"{model_parameters.tmp_path}/yolor_dataset/images/{dataset}"
        labels_dir = xml_dir.replace('images','labels')
        print(f"line161 python xmltotxt.py -c {class_names_path} -xml {xml_dir} -out {labels_dir}")
        bash_command(f"python xmltotxt.py -c {class_names_path} -xml {xml_dir} -out {labels_dir}")
    os.chdir("/workspace/customized/tools") #chdir to training.py

def link_tmp_dataset_to_model_dataset(): 
    # change dataset name
    # tmp_dataset = f"{model_parameters.tmp_path}/your dataset name"
    # symbolic_link_model_dataset = f"{parameters.model_location}/your dataset name"
    tmp_dataset = f"{model_parameters.tmp_path}/yolor_dataset"
    symbolic_link_model_dataset = f"{model_parameters.model_location}/yolor_dataset"
    
    try:
        os.symlink(tmp_dataset, symbolic_link_model_dataset)
    except FileExistsError:
        print(f"model_dataset symbolic_link_root_path File exists! {symbolic_link_model_dataset}")
######################################################

def main():
    # step1 use /workspace/customized/dataset to establish your model dataset structure
    establish_model_dataset_structure()

    # step2 load all xml files and get class_name_list
    class_name_list = get_xml_class_names()
    # step3 get pre_data
    # Yolor: yolor_dataset.yaml
    get_yolor_dataset_yaml(class_name_list)

    # Yolor: class_name.pickle
    get_class_name_pickle(class_name_list)

    # Yolor: yolor_backbone.cfg
    get_yolor_backbone_cfg(class_name_list)

    # Yolor: pretrain_weight.pt
    get_pre_weight()
    
    # step4 Convert the label file to the format required by the model
    convert_dataset_xml_to_yolo()

    # step5 link self.tmp_path dataset to self.model_location dataset
    link_tmp_dataset_to_model_dataset()
    
    # step6 get current model_id
    # model_idx = get_train_model_idx()
    
    # step7 clean current model_id tensorboard files
    clean_previous_tensorboard_files()

if __name__ == '__main__':
    main()
