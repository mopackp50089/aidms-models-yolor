from parent import CVParent
import abc

class ObjectDetection(CVParent, metaclass=abc.ABCMeta):
    def __init__(self, mode):

        super(ObjectDetection, self).__init__(mode)

    @abc.abstractmethod
    def _load_model(self):
        '''
        call this function to load model and save model in self.model .

        Example:
        tf.keras.backend.clear_session()
        model = tf.saved_model.load(self.model_weight_path)
        self.model = model
        '''

    @abc.abstractmethod
    def _infer_model(self):
        '''
        infer image and save figure(visualized), score and class to:
        input:
            self.images             (type: numpy array, (Width, Height, Channel))
        output:
            socres             (type: list, [score1, score2, ...scoreN])
            classes            (type: list, [class_name1, class_name2, ...class_nameN])
            bboxes             (type: list, [[x0, y0, x1, y1], [x0, y0, x1, y1], ...boxN])


        Example:
        self.model.run_on_image(self.images)
        '''
        visualized_img = np.ones((512, 512, 3))
        self.scores = [0.9, 0.8, 0.9]
        self.classes = ['car', 'person', 'cat']
        visualized_img, self.scores, self.classes, self.bboxes  = self.model.run_on_image(self.images)

        # return visualized_img, scores, classes, boxes
        # return scores, classes, bboxes
        
    @abc.abstractmethod
    def load_class_names(self):
        '''
        load your class_map file, such as:
        class_map = ['car', 'dog', 'zebra', ...]
        set:
            self.class_map = ['car', 'dog', 'zebra', ...]
        '''    

if __name__ == '__main__':
    a = ObjectDetection(1,2,3)

