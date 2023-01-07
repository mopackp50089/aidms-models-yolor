from argparse import ArgumentParser
from os import write
from control_weights import bash_command


def write_progress_to_result(model_idx):
    return_code, stdout, stderr = bash_command(
        f'python3 /workspace/customized/tools/get_progress_bar.py > /workspace/aidms/results/model_{model_idx}/progress.txt'
    )

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("model_idx", type=int, help="select model idx")
    args = parser.parse_args()
    write_progress_to_result(args.model_idx)