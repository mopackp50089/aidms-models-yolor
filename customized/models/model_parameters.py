import sys
sys.path.append('/workspace/aidms/lib')
from parameters_class.parameters import Parameters

class Model_Parameters(Parameters):
    def __init__(self,step='training'):
        super().__init__(step)
        #self.obj = obj['results'][select_result_id]['hyperparameters']
        #use self.obj to get the hyperparameter you need (Ex: self.model_name )
        self.model_name = self.obj['training']['default_para']['model']['value'][0]['select_key'] #yolor_p6, yolor_csp

        #model path
        self.model_location = f"/workspace/customized/models/yolor"
        #pre_data path
        self.pre_data_path = f"/workspace/customized/models/pre_weight"
        self.pre_weight = f"{self.pre_data_path}/{self.model_name}.pt" #pre weight path
        self.pre_backbone_cfg = f"{self.pre_data_path}/{self.model_name}.cfg" #backbone path

        #dataset path
        self.aidms_dataset_path = f"/workspace/customized/dataset"
        self.tmp_path = f"/workspace/customized/tmp"
        
        #save result path
        self.results_path = f"/workspace/customized/results/weights"
        #save dataset yaml path
        self.save_dataset_yaml = f"{self.results_path}/yolor_dataset.yaml" #save preprocessing dataset_yaml path
        #save class name map
        self.save_classname_pickle = f"{self.results_path}/class_names.pickle"
        #save backbone_cfg path 
        self.save_backbone_cfg = f"{self.results_path}/{self.model_name}.cfg" #save preprocessing backbone_cfg path
        #save weight path
        self.save_weight_path = f"{self.results_path}/{self.model_name}.pt"
        #save tensorboard
        self.save_tensorboard = f"/workspace/customized/results/tensorboard"

