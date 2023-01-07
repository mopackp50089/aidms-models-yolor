import yaml
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("select_result_id", type=int, help="select model id")
args = parser.parse_args()

with open(r'/workspace/aidms/results/parameters_cluster.yaml', 'r+') as file:
    parameters_cluster = yaml.load(file, Loader=yaml.FullLoader)
    # parameters_cluster['select_result_id'] = f'{args.select_result_id}'
    parameters_cluster['select_result_id'] = args.select_result_id
    file.seek(0)
    file.truncate()
    yaml.dump(parameters_cluster, file)
