#chang yolor cfg filters and classes
import yaml
import os
import argparse

#step1
#yaml_path = 'data/blood.yaml'
cfg_source_path = 'cfg/yolor_p6.cfg'
cfg_target_path = 'cfg/aidms_yolor_p6.cfg'

def change_cfg_filters_classes(yaml_path, cfg_source_path, cfg_target_path):
    #step2
    with open(yaml_path) as f:
        dataMap = yaml.safe_load(f)
    num_classes = len(dataMap['names'])
    num_filters = (num_classes + 5) * 3
    #step3

    with open(file=cfg_target_path, mode='w') as w_f:
        with open(file=cfg_source_path, mode='r') as r_f:
            w_f.write(r_f.read().format(num_classes = str(num_classes), num_filters = str(num_filters)))
    print(f"use {yaml_path} to produce {cfg_target_path} : num_classes={num_classes} num_filters={num_filters}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-yaml', type=str, default='data/blood.yaml', help='data.yaml path')
    opt = parser.parse_args()
    change_cfg_filters_classes(yaml_path=opt.yaml, cfg_source_path=cfg_source_path, cfg_target_path=cfg_target_path)




