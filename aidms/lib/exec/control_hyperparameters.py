import yaml
from argparse import ArgumentParser

def copy_hyperparameters(src=0, dst=1):
# 0 to 5, 0: customized path
    assert src != dst , 'source cannot be as same as destiny.'
    src_path, src_from_customized = ('/workspace/customized/hyperparameters/parameters.yaml', True) if src == 0 else (f'/workspace/aidms/results/parameters_cluster.yaml', False)
    dst_path = '/workspace/customized/hyperparameters/parameters.yaml' if dst == 0 else f'/workspace/aidms/results/parameters_cluster.yaml'
    with open(src_path, 'r') as f:
        src_hyperparameters = yaml.load(f, Loader=yaml.FullLoader)
        if not src_from_customized:
            src_hyperparameters = src_hyperparameters['results'][src]
    with open(dst_path, 'r+') as f:
        dst_hyperparameters = yaml.load(f, Loader=yaml.FullLoader)
        if src_from_customized:
            dst_hyperparameters['results'][dst] = src_hyperparameters
        else:
            dst_hyperparameters = src_hyperparameters
        f.seek(0)
        f.truncate()
        documents = yaml.dump(dst_hyperparameters, f)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("src", type=int, help="source")
    parser.add_argument("dst", type=int, help="destiny")
    args = parser.parse_args()
    copy_hyperparameters(args.src, args.dst)

