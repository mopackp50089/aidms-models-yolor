import glob
import math
import os
import yaml

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import sys
sys.path.append('/workspace/aidms/lib/')
# from parameters_class.parameters import Parameters
# yolor
sys.path.append('/workspace/customized/models/')
from model_parameters import Model_Parameters
model_parameters = Model_Parameters(step='training')

class ProgressBar():  
    def _get_current_iter(self):
        try:
            #---------------modify---------------
            #Change to <your tensorboard events.out.tfevents*>
            # Ex: 
            # glob.glob(f'/workspace/result/{self.select_result_id}/<your tensorboard events.out.tfevents*>')[0]
            path_to_events_file = glob.glob(f'{model_parameters.save_tensorboard}/events.out.tfevents*')[0]
            #------------------------------------
            event_acc = EventAccumulator(path_to_events_file)
            event_acc.Reload()
            #---------------modify---------------
            #Change to <your tensorboard Tags>
            # Ex: 
            # event_acc.Scalars(<your tensorboard Tags>)
            for e in event_acc.Scalars('train/cls_loss'): 
                pass
      
        except (IndexError, KeyError): # IndexError: # tensorboard has not built yet. KeyError: tf has not assign key yet.
            return 0

        return e.step + 1
    
    def _get_total_iter(self):
        with open('/workspace/customized/hyperparameters/parameters.yaml', 'r') as f:
            hyperparameters = yaml.load(f, Loader=yaml.FullLoader)
        total_iter = hyperparameters['hyperparameters']['training']['default_para']['epoch']['value']

        return total_iter

    def get_progress_percentage(self):
        total_iter = self._get_total_iter()
        if total_iter==0:
            print(0)
        else:
            current_iter = self._get_current_iter()
            progress_percentage = current_iter/total_iter*100
            print(math.ceil(progress_percentage))


if __name__ == '__main__':
    # parser = ArgumentParser()
    # parser.add_argument("select_result_id", type=str, help="select model id")
    # args = parser.parse_args()
    # progress_bar = ProgressBar(args.select_result_id)
    progress_bar = ProgressBar()
    progress_bar.get_progress_percentage()
