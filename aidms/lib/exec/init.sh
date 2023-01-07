#!/bin/bash
num_result=5
start=1
model_path=/workspace/aidms/results
rm -r /workspace/aidms/tmp/*
mkdir /workspace/aidms/tmp/metric
rm -r /workspace/customized/tmp/*
rm /workspace/customized/dataset/train/*
rm /workspace/customized/dataset/validation/*
rm /workspace/customized/dataset/test/*
rm /workspace/customized/results/weights/*.pickle
rm /workspace/customized/results/weights/*.pth
rm /workspace/customized/results/weights/onnx/*.onnx

# for (( c=$start; c<=$num_result; c++ ))
# do
#     echo "$c"
#     each_model_path=${model_path}/model_${c}/
#     echo "$each_model_path"
#     find ${each_model_path} ! -name 'training.log' -type f,l -exec rm -f {} +
# done
find ${model_path} ! -name '.gitkeep' -type f,l -exec rm -f {} +


if [ -z $1 ]
then
    python3 create_parameters_cluster.py 5
    python3 set_select_model_id.py 2
    python3 split_dataset.py 
    echo "split dataset!"
else
    echo "No split dataset!"
fi
