import pickle
import glob
import os
import numpy as np
import pandas as pd
import json
from argparse import ArgumentParser
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

def main(model_idx):
    file_path = '/workspace/aidms/tmp/metric/annotations_res.json'

    with open(file_path, 'r') as f:
        raw_results = f.readlines()[0]
    raw_results = json.loads(raw_results)

    with open('/workspace/aidms/tmp/class_names.pickle', 'rb') as f:
        class_name_list = pickle.load(f)
        class_name_list.remove('__ignore__')

    y_true, y_pred = [], []
    for raw_result in raw_results:
        image_name = list(raw_result.keys())[0]
        image_label_file_name = os.path.join('/workspace/customized/dataset/test', f'{os.path.splitext(image_name)[0]}.txt')
        with open(image_label_file_name, 'r') as f:
            image_gt_class = f.read().strip()
        image_gt_class_idx = class_name_list.index(image_gt_class)
        image_pd_class_idx = raw_result[image_name]
        y_true.append(image_gt_class_idx)
        y_pred.append(image_pd_class_idx)


    

    # def get_test_dataset_img_names_and_classes():
    # classes_path = glob.glob(f'/workspace/tmp/test/*')
    
    # for class_path in classes_path:
    #     one_class_imgs = glob.glob(f'{class_path}/*')
    #     for one_class_img in one_class_imgs:
    #         img_name = os.path.basename(one_class_img)
    #         img_class = os.path.basename(class_path)

    #         img_class_idx = class_name_list.index(img_class)
    #         predict_img_class_idx = results[img_name]

    #         y_true.append(img_class_idx)
    #         y_pred.append(predict_img_class_idx)

    class_name_idx = [class_idx for class_idx in range(len(class_name_list))]
    conf_matrix = confusion_matrix(y_true, y_pred, labels=class_name_idx)
    conf_matrix_df = pd.DataFrame(conf_matrix, columns=class_name_list)
    conf_matrix_df.insert(loc=0, column='Confusion matrix ( True \ predicted )', value=class_name_list)
    conf_matrix_df.to_csv(f'/workspace/aidms/results/model_{model_idx}/confusion_matrix.csv', index=False)
    p_score = precision_score(y_true, y_pred, labels=class_name_idx, average=None)
    r_score = recall_score(y_true, y_pred, labels=class_name_idx, average=None)
    f_score = f1_score(y_true, y_pred, labels=class_name_idx, average=None)
    score_df = pd.DataFrame({'F1 score':f_score, 'Precision':p_score, 'Recall':r_score})
    score_df.insert(loc=0, column='Class', value=class_name_list)
    score_df = score_df.round(decimals=3)
    score_df.to_csv(f'/workspace/aidms/results/model_{model_idx}/score.csv', index=False)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("select_result_id", type=str, help="select model id")
    args = parser.parse_args()
    main(args.select_result_id)
