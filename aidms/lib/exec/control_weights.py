import yaml
from argparse import ArgumentParser
import subprocess

def bash_command(cmd):
    result = subprocess.Popen(['/bin/bash', '-c', cmd], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    std_output, std_error = result.communicate()
    # print(text)
    return result.returncode, std_output.decode("utf-8"), std_error.decode("utf-8")

def copy(src, dist):
    returncode, stdout, stderr = bash_command(
        f'cp -r {src}/* {dist}'
    )

def copy_weight(src=0, dst=1):
# 0 to 5, 0: customized path
    assert src != dst , 'source cannot be as same as destiny.'
    src_path = '/workspace/customized/results/weights' if src == 0 else f'/workspace/aidms/results/model_{src}/weight'
    dst_path = '/workspace/customized/results/weights' if dst == 0 else f'/workspace/aidms/results/model_{dst}/weight'
    copy(src_path, dst_path)

def delete_weight(src=0):
    src_path = '/workspace/customized/results/weights' if src == 0 else f'/workspace/aidms/results/model_{src}/weight'
    returncode, stdout, stderr = bash_command(f'rm -f {src_path}/*')

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("src", type=int, help="source")
    parser.add_argument("dst", type=int, help="destiny")
    args = parser.parse_args()
    copy_weight(args.src, args.dst)

