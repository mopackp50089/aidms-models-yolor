import shutil
import subprocess
import os
from argparse import ArgumentParser
import yaml
from control_hyperparameters import copy_hyperparameters
from control_weights import copy_weight, bash_command

class CommitModel(object):
    def __init__(self, select_result_id):
        self.select_result_id = select_result_id

    def commit_model(self):
        self._copy_model_weight()
        self._copy_parameters_yaml()
        self._delete_tmp()

    def _copy_model_weight(self):
        copy_weight(self.select_result_id, 0)

    def _copy_parameters_yaml(self):
        copy_hyperparameters(self.select_result_id, 0)

    def _delete_tmp(self):
        # bash_command('/workspace/init.sh svn')
        bash_command('rm -r /workspace/customized/tmp/*')
            


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("commit_result_id", type=int, help="select model id")
    args = parser.parse_args()

    cm = CommitModel(args.commit_result_id)
    cm.commit_model()
