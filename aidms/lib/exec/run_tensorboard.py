import os
from argparse import ArgumentParser

if __name__ == '__main__':
    # kill previous tensorboard server first.
    os.system("kill -9 $(ps aux | grep 'tensorboard --logdir' | grep -v 'grep' | awk {{'print$2'}})")
    parser = ArgumentParser()
    parser.add_argument("result_id", type=int, help="set result_id, use 0 to view all result of IDs")
    parser.add_argument("training_status", type=int, default=0, help="0: non-training; 1: training. So tensorboard file path is different.")
    args = parser.parse_args()
    if args.result_id == 0:
        os.system(f'tensorboard --logdir /workspace/aidms/results --bind_all &')
    elif args.training_status:
        os.system(f'tensorboard --logdir /workspace/customized/results/tensorboard --bind_all &')
    else:
        os.system(f'tensorboard --logdir /workspace/aidms/results/model_{args.result_id}/tensorboard --bind_all &')
    