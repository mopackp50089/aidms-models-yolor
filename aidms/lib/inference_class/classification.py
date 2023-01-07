from parent import CVParent
import abc


class Classification(CVParent, metaclass=abc.ABCMeta):
    def __init__(self, model_weight_path, mode):

        super(Classification, self).__init__(model_weight_path, mode)

    @abc.abstractmethod
    def _load_model(self):
        '''
        call this function to load model.

        Example:
        tf.keras.backend.clear_session()
        model = tf.saved_model.load(self.model_weight_path)
        return model
        '''

    @abc.abstractmethod
    def _infer_model(self):
        '''
        infer image and save figure(visualized), score and class to:
        input:
            self.images             (type: numpy array, (Width, Height, Channel))
        output:
            self.socres             (type: list, [score1, score2, ...scoreN])
            self.classes            (type: list, [class_name1, class_name2, ...class_nameN])

        Example:
        self.model.run_on_image(self.images)
        '''

        scores = [0.9, 0.8, 0.9]
        classes = ['car', 'person', 'cat']
        scores, classes  = self.model.run_on_image(self.images)
        self.scores = [0.9, 0.8, 0.9] # confidence of each object
        self.classes = ['car', 'person', 'cat'] # class name of each object
        
    def get_model_result_for_client(self):
        self.b64_images = self.nparr2b64(self.visual_images)
        result = {
            'scores': self.scores, 
            'classes': self.classes,
            }
        return result

    

a = Classification(image_size=512, batch_size=1)