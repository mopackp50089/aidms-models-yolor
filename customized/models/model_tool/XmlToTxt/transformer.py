import os

from objectmapper import ObjectMapper
from reader import Reader


class Transformer(object):
    def __init__(self, xml_dir, out_dir, class_file):
        self.xml_dir = xml_dir
        self.out_dir = out_dir
        self.class_file = class_file

    def transform(self):
        print(self.xml_dir)
        reader = Reader(xml_dir=self.xml_dir)
        xml_files = reader.get_xml_files()
        classes = reader.get_classes(self.class_file)
        print('line17',type(classes),classes)
        object_mapper = ObjectMapper()
        annotations = object_mapper.bind_files(xml_files, xml_dir=self.xml_dir)
        print('line21',xml_files)
        #harry
        #self.get_classes_names(annotations)
        self.write_to_txt(annotations, classes)

    def write_to_txt(self, annotations, classes):
        print('line27 total image:', len(annotations))
        for annotation in annotations:
            print('line28',annotation)
            output_path = os.path.join(self.out_dir, self.darknet_filename_format(annotation.filename))
            if not os.path.exists(os.path.dirname(output_path)):
                os.makedirs(os.path.dirname(output_path))
            with open(output_path, "w+") as f:
                print('line31',output_path)
                f.write(self.to_darknet_format(annotation, classes))

    def to_darknet_format(self, annotation, classes):
        result = []
        for obj in annotation.objects:
            if obj.name not in classes:
                print("Please, add '%s' to classes.txt file." % obj.name)
                exit()
            print('line40',obj)
            x, y, width, height = self.get_object_params(obj, annotation.size)
            result.append("%d %.6f %.6f %.6f %.6f" % (classes[obj.name], x, y, width, height))
        return "\n".join(result)

    @staticmethod
    def get_object_params(obj, size):
        image_width = 1.0 * size.width
        image_height = 1.0 * size.height

        box = obj.box
        absolute_x = box.xmin + 0.5 * (box.xmax - box.xmin)
        absolute_y = box.ymin + 0.5 * (box.ymax - box.ymin)
        absolute_width = box.xmax - box.xmin #box_width
        absolute_height = box.ymax - box.ymin #box_height

        x = absolute_x / image_width
        y = absolute_y / image_height
        width = absolute_width / image_width
        height = absolute_height / image_height
        print('line50 width,height',size.width,size.height)
        print('line51 box_xmin,box_xmax',box.xmin,box.xmax)
        print('line52 box_ymin,box_ymax',box.ymin,box.ymax)
        print('line53 center',absolute_x,absolute_y)

        return x, y, width, height

    @staticmethod
    def darknet_filename_format(filename):
        pre, ext = os.path.splitext(filename)
        return "%s.txt" % pre