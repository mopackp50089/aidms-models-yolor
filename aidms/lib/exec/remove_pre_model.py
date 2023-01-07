import subprocess
from argparse import ArgumentParser
from control_weights import delete_weight

def bash_command(cmd):
    result = subprocess.Popen(['/bin/bash', '-c', cmd], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    std_output, std_error = result.communicate()
    # print(text)
    return result.returncode, std_output.decode("utf-8"), std_error.decode("utf-8")


def remove_premodel_by_id(select_result_id):
    model_paths = [
        f'/workspace/aidms/results/model_{select_result_id}', 
        '/workspace/aidms/tmp',
        '/workspace/customized/results',
        '/workspace/customized/tmp'
    ]
    for model_path in model_paths:
        returncode, stdout, stderr = bash_command(f"find {model_path} ! -name 'training.log' -type f,l -exec rm -f {{}} +")
        print('Remove previous model result: ', 'Process successfully!'if not returncode else 'Warning! Process unsuccessfully')
        if not returncode:
            print(f'{model_path} has been removed. ')
        else:
            print(f'{model_path} dont have any file to remove. ')

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("select_result_id", type=str, help="set remove model ID")
    args = parser.parse_args()    
    remove_premodel_by_id(args.select_result_id)
