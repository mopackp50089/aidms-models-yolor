import os
import json
from os import listdir, getcwd
from os.path import join

classes = ["NG"]
#raise 'change classes'

#box form[x,y,w,h]
def convert(size,box):
    dw = 1./size[0]
    dh = 1./size[1]
    x = box[0]*dw
    y = box[1]*dh
    w = box[2]*dw
    h = box[3]*dh
    return (x,y,w,h)

def convert_annotation(json_file_path,save_labels_path):
	try:
		os.makedirs(save_labels_path)
	except:
		print('Directory is exit!')
	with open(json_file_path,'r') as f:
		data = json.load(f)
	class_name_set = set()
	for item in data['images']: #item:{'height': 2056, 'width': 2464, 'id': 1, 'file_name': '/workspace/tmp/yolor_dataset/images/test/NG-00057_A_S_H.png'}
		image_id = item['id']
		file_name = os.path.basename(item['file_name'])
		width = item['width']
		height = item['height']
		#抓出對應image_id的data['annotations']
		value = filter(lambda item1: item1['image_id'] == image_id,data['annotations'])#某image_id的data['annotations']
		outfile = open(save_labels_path+'%s.txt'%(file_name[:-4]), 'a+')
		for item2 in value:
			category_id = item2['category_id']
			value1 = filter(lambda item3: item3['id'] == category_id,data['categories'])
			for v in value1:
				name = v['name']#{'supercategory': 'Cancer', 'id': 1, 'name': 'NG'}
				class_name_set.add(name)
			#name = value1[0]['name'] <- 原code的Bug，filter object不能filter[0]
			class_id = classes.index(name)
			#print('class_id',class_id)
			box = item2['bbox']
			bb = convert((width,height),box)
			outfile.write(str(class_id)+" "+" ".join([str(a) for a in bb]) + '\n')
		outfile.close()
	return list(class_name_set)
# if __name__ == '__main__':
# 	#dir in there
# 	print('source dir:',os.getcwd())
# 	#change dir path
# 	fd = os.open( "/workspace/models/yolor/Data_Transform_tool", os.O_RDONLY )
# 	os.fchdir(fd)
# 	print('change dir:',os.getcwd())
   
# 	convert_annotation('../aidms_dataset/annotations/instances_val2017.json','../aidms_dataset/labels/aidms_val/')
# 	convert_annotation('../aidms_dataset/annotations/instances_train2017.json','../aidms_dataset/labels/aidms_train/')
	
	
