import yaml
from argparse import ArgumentParser
import os
import subprocess
import json

def bash_command(cmd):
    result = subprocess.Popen(['/bin/bash', '-c', cmd], stdout=subprocess.PIPE)
    text = result.communicate()[0]
    # print(text)
    return result.returncode, len(text)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("num_result", type=int, help="number of result")
    args = parser.parse_args()

    with open(r'/workspace/customized/hyperparameters/parameters.yaml', 'r+') as file:
        fruits_list = yaml.load(file, Loader=yaml.FullLoader)

    parameters_cluster = {'results':dict()}
    
    # copy parameters.yaml 'num_result' times to be aggregate into parameters_cluster.yaml.
    for result_id in range(1, args.num_result+1): # +1 is default
        # key_str = f'{str(result_id)}'
        key_str = result_id
        idx = {'status':'null'}
        if result_id == 0:
            key_str = 'default'
            idx = {}
        idx.update(fruits_list)
        parameters_cluster['results'].update({key_str:idx})

        # create /workspace/result/{i}
        [bash_command(f'mkdir -p /workspace/aidms/results/model_{result_id}/{folder}') for folder in ('log', 'tensorboard', 'weight')]
        
        returncode, _ = bash_command(f'echo 0 > /workspace/aidms/results/model_{result_id}/progress.txt')
        returncode, _ = bash_command(f'touch /workspace/aidms/results/model_{result_id}/log/training.log')
        # copy model weight from /workspace/weight into /workspace/result/*/model_weight/
        returncode, _ = bash_command(f'cp -r /workspace/customized/results/weights/* /workspace/aidms/results/model_{result_id}/weight/')
        # print('returncode', returncode)
    bash_command(f'mkdir /workspace/aidms/results/upload_images')
    bash_command(f'mkdir /workspace/aidms/tmp/metric')
    parameters_cluster.update({'select_result_id':1})
    yaml.Dumper.ignore_aliases = lambda *args : True
    with open(r'/workspace/aidms/results/parameters_cluster.yaml', 'w+') as file:
        file.seek(0)
        file.truncate()
        yaml.dump(parameters_cluster, file, default_flow_style=False)

    # create user upload images directory
    # returncode, _ = bash_command(f'mkdir -p /workspace/result/upload_images')

    # create split_dataset.json
    split_dataset_json = {'train': [], 'validation': [], 'test': []}
    with open(os.path.join('/workspace', 'aidms', 'results', 'split_dataset.json'), 'w') as f:
        f.write(json.dumps(split_dataset_json))
