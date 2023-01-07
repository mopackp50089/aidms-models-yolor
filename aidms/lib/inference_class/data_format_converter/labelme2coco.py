import argparse
import collections
import datetime
import glob
import json
import os
import os.path as osp
import sys
import uuid

import numpy as np
import PIL.Image

import labelme

import pickle

try:
    import pycocotools.mask
except ImportError:
    print('Please install pycocotools:\n\n    pip install pycocotools\n')
    sys.exit(1)
import ntpath

def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('input_dir', help='input annotated directory')
    parser.add_argument('output_dir', help='output dataset directory')
    parser.add_argument('--labels', help='labels file', required=True)
    #david modify:
    # parser.add_argument('--image_size', type=int, help='set image size to reset xyxy', required=True)
    parser.add_argument('--image_size_w', type=int, help='set image size to reset xyxy', required=True)
    parser.add_argument('--image_size_h', type=int, help='set image size to reset xyxy', required=True)
    args = parser.parse_args()

    if not osp.exists(args.output_dir):
        print('Output directory already exists:', args.output_dir)
        # sys.exit(1)
        os.makedirs(args.output_dir)
        os.makedirs(osp.join(args.output_dir, 'JPEGImages'))
        print('Creating dataset:', args.output_dir)

    now = datetime.datetime.now()

    data = dict(
        info=dict(
            description=None,
            url=None,
            version=None,
            year=now.year,
            contributor=None,
            date_created=now.strftime('%Y-%m-%d %H:%M:%S.%f'),
        ),
        licenses=[dict(
            url=None,
            id=0,
            name=None,
        )],
        images=[
            # license, url, file_name, height, width, date_captured, id
        ],
        type='instances',
        annotations=[
            # segmentation, area, iscrowd, image_id, bbox, category_id, id
        ],
        categories=[
            # supercategory, id, name
        ],
    )
    #david modify
    class_name_to_id = {}
    # for i, line in enumerate(open(args.labels).readlines()):
    with open(args.labels, 'rb') as f:
        class_name_list = pickle.load(f)
    for i, line in enumerate(class_name_list):
        class_id = i - 1  # starts with -1
        class_name = line
        if class_id == -1:
            assert class_name == '__ignore__'
            continue
        class_name_to_id[class_name] = class_id
        data['categories'].append(dict(
            supercategory=None,
            id=class_id,
            name=class_name,
        ))

    out_ann_file = osp.join(args.output_dir, 'annotations.json')
    label_files = glob.glob(osp.join(args.input_dir, '*.json'))
    
    for image_id, label_file in enumerate(label_files):
        # print('Generating dataset from:', label_file)
        with open(label_file) as f:
            label_data = json.load(f)

        base = osp.splitext(osp.basename(label_file))[0]
        out_img_file = osp.join(
            args.output_dir, 'JPEGImages', base + '.jpg'
        )

        #img_file = osp.join(
        #    osp.dirname(label_file), label_data['imagePath'].replace('bmp','jpg')
        #)
        img_file = osp.join(
            osp.dirname(label_file), ntpath.basename(label_data['imagePath'])
        )

        img = np.asarray(PIL.Image.open(img_file))
        # PIL.Image.fromarray(img).save(out_img_file)
        data['images'].append(dict(
            license=0,
            url=None,
            file_name=osp.relpath(out_img_file, osp.dirname(out_ann_file)),
            # david modify:
            # height=img.shape[0],
            # width=img.shape[1],
            height=args.image_size_h,
            width=args.image_size_w,
            date_captured=None,
            # david modify
            # id=image_id
            id=base
        ))

        masks = {}                                     # for area
        segmentations = collections.defaultdict(list)  # for segmentation
        for shape in label_data['shapes']:
            points = shape['points']
            if len(points) <= 2:
                print('points in mask should be larger than 2 !, we will skip this mask of file !')
                continue
            label = shape['label']
            group_id = shape.get('group_id')
            shape_type = shape.get('shape_type')
            mask = labelme.utils.shape_to_mask(
                img.shape[:2], points, shape_type
            )

            if group_id is None:
                group_id = uuid.uuid1()

            instance = (label, group_id)

            if instance in masks:
                masks[instance] = masks[instance] | mask
            else:
                masks[instance] = mask

            points = np.asarray(points).flatten().tolist()
            segmentations[instance].append(points)
        segmentations = dict(segmentations)

        for instance, mask in masks.items():
            cls_name, group_id = instance
            if cls_name not in class_name_to_id:
                continue
            cls_id = class_name_to_id[cls_name]

            mask = np.asfortranarray(mask.astype(np.uint8))
            mask = pycocotools.mask.encode(mask)
            area = float(pycocotools.mask.area(mask))
            bbox = pycocotools.mask.toBbox(mask).flatten().tolist()
            # david modify
            bbox[0] = bbox[0]*args.image_size_w/img.shape[1]
            bbox[2] = bbox[2]*args.image_size_w/img.shape[1]
            bbox[1] = bbox[1]*args.image_size_h/img.shape[0]
            bbox[3] = bbox[3]*args.image_size_h/img.shape[0]
            bbox_area = bbox[2] * bbox[3]

            # seg_one_inst = np.asarray(segmentations[instance].copy())
            # seg_one_inst[0][::2] = seg_one_inst[0][::2]*args.image_size/img.shape[1]
            # seg_one_inst[0][1::2] = seg_one_inst[0][1::2]*args.image_size/img.shape[0]
            # segmentations[instance] = seg_one_inst.tolist()
            # end
            data['annotations'].append(dict(
                id=len(data['annotations'])+1, # I found that id should start from 1!
                #david modify
                # image_id=image_id,
                image_id=base,
                category_id=cls_id,
                segmentation=segmentations[instance],
                # segmentation=[[bbox[0], bbox[1], bbox[2]+bbox[0], bbox[3]+bbox[1]]],
                area=bbox_area,
                bbox=bbox,
                iscrowd=0,
            ))

    with open(out_ann_file, 'w') as f:
        json.dump(data, f)


if __name__ == '__main__':
    main()
