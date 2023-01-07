import matplotlib.pyplot as plt
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import numpy as np
from argparse import ArgumentParser
import sys
from io import StringIO
import pandas as pd
import re
import os

def coco_op2csv(coco_output_string):
    
    coco_output_string = coco_output_string.split('\n')[:-1]
    
    df = pd.DataFrame(columns = ['AP/AR', 'IoU', 'area', 'maxDets', 'AP score'])
    replace_signs = ('[', ']', '|')
    replace_signs2 = ('IoU', 'area', '=', '@')
    new_line = []
    for line in coco_output_string:
        for replace_sign in replace_signs:
            line = line.replace(replace_sign, ',')

        for replace_sign in replace_signs2:
            line = line.replace(replace_sign, '')

        line = line.replace(' ', '').split(',')
        df = df.append(pd.Series(line, index=df.columns), ignore_index=True)
    
    # return df[1::12] #only AP not AR, all class in one df
    return df

def get_mAP50(string):
    string = string.split('\n')[1] # get row of mAP 50 .
    regex = re.compile('= (\d\.\d+)')
    match = regex.search(string)

    if match == None:
        mAP = 'Nan'
    else:
        mAP = float(match.group(1))

    return mAP

def get_all_mAP(string):
    all_mAP = re.findall('= (\d\.\d+)', string)
    
    return all_mAP

def evaluate_mAP(cocoEval):
    
    cocoEval.evaluate()
    cocoEval.accumulate()
    stdout_ = sys.stdout 
    stream = StringIO()
    sys.stdout = stream
    cocoEval.summarize() 
    sys.stdout = stdout_ 
    variable = stream.getvalue()

    return variable

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("select_result_id", type=str, help="select model id")
    args = parser.parse_args()
    annType = ['segm','bbox','keypoints']
    annType = annType[1]      #specify type here
    prefix = 'person_keypoints' if annType=='keypoints' else 'instances'

    annFile = '/workspace/aidms/tmp/metric/annotations.json'
    # annFile = '/workspace/instances_val2014.json'
    cocoGt=COCO(annFile)

    resFile = '/workspace/aidms/tmp/metric/annotations_res.json'
    # resFile = '/workspace/instances_fakebbox1.json'
    res_txt = open('/workspace/aidms/tmp/metric/annotations_res.json', 'r').read().strip()
    class_mAP = dict()
    mAPs_name = ('AP', 'AP|50', 'AP|75', 'AP|S', 'AP|M',  'AP|L')
    class_mAP['Class \ AP'] = ['All']
    class_mAP.update({mAP_name : [] for mAP_name in mAPs_name})

    if res_txt == '[]': # check if annotations_res.json is empty, just output nan value for each class
        catIds = list(cocoGt.cats.keys())
        for catId in catIds:
            class_mAP['Class \ AP'].append(cocoGt.cats[catId]['name'])
        for mAP_idx in range(len(mAPs_name)): class_mAP[mAPs_name[mAP_idx]].extend(['nan'] * (len(catIds) + 1) )
    else:
        cocoDt=cocoGt.loadRes(resFile)

        # running evaluation
        cocoEval = COCOeval(cocoGt,cocoDt,annType)
        # cocoEval.params.catIds = [65]
        # cocoEval.params.imgIds = [776]
        # cocoEval.params.catIds = [4]
        # cocoEval.params.imgIds = [0]
        # cocoEval.params.imgIds  = imgIds

        
        variable = evaluate_mAP(cocoEval)
        # only mAP50
        # mAP = get_mAP50(variable)

        # all mAP:
        all_mAP = get_all_mAP(variable)
        
        for mAP_idx in range(len(mAPs_name)): class_mAP[mAPs_name[mAP_idx]].append(all_mAP[mAP_idx])
        for one_catId in cocoEval.params.catIds:
            cocoEval.params.catIds = [one_catId]
            variable = evaluate_mAP(cocoEval)
            # only mAP50
            # mAP = get_mAP50(variable)
            # class_mAP[cocoGt.cats[one_catId]['name']] = [mAP]

            # all mAP:
            class_mAP['Class \ AP'].append(cocoGt.cats[one_catId]['name'])
            all_mAP = get_all_mAP(variable)
            
            if not all_mAP:
                for mAP_idx in range(len(mAPs_name)): class_mAP[mAPs_name[mAP_idx]].append('nan')
            else:
                for mAP_idx in range(len(mAPs_name)): class_mAP[mAPs_name[mAP_idx]].append(all_mAP[mAP_idx])
        print(class_mAP)
            # mAP_df = coco_op2csv(variables)

    mAP_df = pd.DataFrame(data=class_mAP)
    mAP_df.to_csv(f'/workspace/aidms/results/model_{args.select_result_id}/mAP.csv', index=False)

