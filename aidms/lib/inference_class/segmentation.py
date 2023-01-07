from parent import CVParent
import abc

class Segmentaion(CVParent, metaclass=abc.ABCMeta):
    def __init__(self, mode):

        super(Segmentaion, self).__init__(mode)

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
    def _infer_model(self, images):
        '''
        infer image and save figure(visualized), score and class to:
        input:
            images             (type: numpy array, (Batch, Width, Height, Channel))
        output:
            self.socres             (type: list, [score1, score2, ...scoreN])
            self.classes            (type: list, [class_name1, class_name2, ...class_nameN])
            self.bboxes             (type: list, [[x0, y0, x1, y1], [x0, y0, x1, y1], ...boxN])
            self.masks              ((type: list, [[w,h], [w,h], ...mask_N]), element of [w,h] is 1 or 0)

        Example:
        self.model.run_on_image(images)
        '''
        visualized_img = np.ones((512, 512, 3))
        scores = [0.9, 0.8, 0.9]
        classes = ['car', 'person', 'cat']
        self.scores, self.classes, self.masks  = self.model.run_on_image(self.images)

        # # return visualized_img, scores, classes, bboxes
        # return scores, classes, bboxes, masks
        
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

