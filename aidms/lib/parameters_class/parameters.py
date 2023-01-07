import yaml 
# from aidms.parameters_class.hyperparameters_to_attribute import AttributeDict


class Parameters():
    _inference_yaml_path = '/workspace/customized/hyperparameters/parameters.yaml'
    _training_yaml_path = '/workspace/aidms/results/parameters_cluster.yaml'

    def __init__(self, step='training', result_id=0):
        self.obj, self.model_type, self.select_result_id = self.load_yaml(step, result_id=result_id)
        self.skip_keys = ['description'] # we will not check the type in these keys.
        self.use = None

    # def __repr__(self):
    #     return f'The .yaml has: {list(self.hyperpara)} parameters!'

    # when select fist index rule:
    # def __getitem__(self, hprs):
    #     self.use = None
    #     obj = self.obj.copy()
    #     for hpr in hprs:
    #         obj = obj[hpr]
    #     obj = self._check_list_or_dict(obj)
    #     # obj = AttributeDict(**obj)
    #     return obj, self.use

    def __getitem__(self, hprs):
        self.use = None
        obj = self.obj.copy()
        for hpr in hprs:
            obj = obj[hpr]
        obj = self._check_list_or_dict(obj)
        if isinstance(obj, dict):
            obj_use = list(obj.keys())[0]
        else:
            obj_use = None
        # obj = AttributeDict(**obj)
        return obj, obj_use

    def _check_use(self, obj):
        if isinstance(obj, dict):
            keys = obj.keys()
            # print('key', keys)
            if 'disabled' in keys:
                self.use = False
                obj[False] = obj.pop('disabled')
            elif 'enabled' in keys:
                self.use = True
                obj[True] = obj.pop('enabled')
            # print(self.use)

        
    
    def _check_list_or_dict(self, obj):
        if isinstance(obj, list):
             obj = self._select_first_idx(obj)
            #  print(obj)
        elif isinstance(obj, dict):
            # print(obj)
            obj = self._scan_all_keys(obj)
        # print(obj)
        return obj

    # when select fist index rule:
    # def _select_first_idx(self, obj):
    #     # print(obj)
    #     obj = obj[0]
    #     self._check_use(obj)
    #     obj = self._check_list_or_dict(obj)
    #     # print(obj)
    #     return obj

    def _select_first_idx(self, obj):
        o = obj[0]
        if isinstance(o, int) or isinstance(o, float):
            # print(o, 'reading "limit range"')
            obj = obj[0]
        else:
            select_key = o['select_key']
            for o in obj:
                if isinstance(o, dict):
                    if list(o.keys())[0] == select_key:
                        obj = o
                        break
                else:
                    if o == select_key:
                        obj = o
                        break
        self._check_use(obj)
        obj = self._check_list_or_dict(obj)
        # print(obj)
        return obj
    
    def _scan_all_keys(self, obj):
        # print('a', obj.keys())
        keys = obj.copy().keys()
        for key in keys:
            if key in self.skip_keys:
                obj.pop(key)
                continue
            elif key == 'value' and not isinstance(obj['value'],list):
                obj = obj['value']
                break
            elif key == 'value' and isinstance(obj['value'],list):
                obj = obj['value']
                obj = self._check_list_or_dict(obj)
                break
            sub_obj = obj[key]
            sub_obj = self._check_list_or_dict(sub_obj)
            obj[key] = sub_obj
        
        return obj
            
    def add_skip_keys(self, key):
        self.skip_keys.append(key)

                
    def load_yaml(self, step, result_id=0):
        if step == 'training':
            try:
                obj, model_type, select_result_id = self._load_yaml_from_cluster(result_id)
            except FileNotFoundError:
                obj, model_type, select_result_id = self._load_yaml_from_single()
        else:
            obj, model_type, select_result_id = self._load_yaml_from_single()

        return obj, model_type, select_result_id

    def _load_yaml_from_cluster(self, result_id):
        with open(self._training_yaml_path, 'r+') as file:
            obj = yaml.load(file, Loader=yaml.FullLoader)
            if result_id:
                select_result_id = result_id
            else:
                select_result_id = obj['select_result_id']
            model_type = obj['results'][select_result_id]['model_type']
            self.status = obj['results'][select_result_id]['status']
            obj = obj['results'][select_result_id]['hyperparameters']
        
        return obj, model_type, select_result_id

    def _load_yaml_from_single(self):
        with open(self._inference_yaml_path, 'r+') as file:
            obj = yaml.load(file, Loader=yaml.FullLoader)
            select_result_id = 'default'
            model_type = obj['model_type']
            self.status = None
            obj = obj['hyperparameters']

        return obj, model_type, select_result_id

        
    
if __name__ == '__main__':
    a = Parameters(step='training')
    a.add_skip_keys('limit_range')
    b, c = a['training', 'advanced_para', 'image_crop']
    # b, c = a['training', 'advanced_para', 'image_size']
    # b, c = a['training', 'advanced_para', 'image_rotate']
    # b, c = a['training', 'advanced_para', 'image_flip']
    # b, c = a ['inference', 'advanced_para', 'DETECTION_MAX_INSTANCES']
    # with open(r'train.yaml', 'w') as file:
    #     documents = yaml.dump(b, file)

    print(1)
