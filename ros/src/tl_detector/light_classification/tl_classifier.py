from styx_msgs.msg import TrafficLight
import datetime
import cv2
import sys
import os

import tensorflow as tf

PATH_TF_MODELS_RESEARCH = "/home/student/github/models/research"
PATH_TF_MODELS_SLIM = "/home/student/github/models/research/slim"
PATH_TF_MODELS_OBJECT_DETECTION = "/home/student/github/models/research/object_detection"

sys.path.append(PATH_TF_MODELS_RESEARCH)
sys.path.append(PATH_TF_MODELS_SLIM)
sys.path.append(PATH_TF_MODELS_OBJECT_DETECTION)
from utils import label_map_util
from utils import visualization_utils as vis_util

FILE_PREFIX_IMG = "IMG_"
DIR_DATA = "DATA/"


PATH_TRAINED_GRAPH = "/home/student/github/models/research/object_detection/ssd_mobilenet_v1_coco_2017_11_17/frozen_inference_graph.pb"
PATH_LABEL_MAP = "/home/student/github/models/research/object_detection/data/mscoco_label_map.pbtxt"
NUM_CLASSES = 90

class TLClassifier(object):
    def __init__(self):
        #TODO load classifier
        #self.append_required_modules()
        self.load_model()
        self.load_label_map()
        pass

    def append_required_modules(self):
        sys.path.append(PATH_TF_MODELS_RESEARCH)
        sys.path.append(PATH_TF_MODELS_SLIM)
        sys.path.append(PATH_TF_MODELS_OBJECT_DETECTION)
        from utils import label_map_util
        from utils import visualization_utils as vis_util
        print(sys.path)

    def load_model(self):
        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            graph_def = tf.GraphDef()
            with tf.gfile.GFile(PATH_TRAINED_GRAPH, 'rb') as fid:
                serialized_graph = fid.read()
                graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(graph_def, name='')

    def load_label_map(self):
        self.label_map = label_map_util.load_labelmap(PATH_LABEL_MAP)
        self.categories = label_map_util.convert_label_map_to_categories(self.label_map,
                            max_num_classes=NUM_CLASSES, use_display_name=True)
        self.category_index = label_map_util.create_category_index(self.categories)

    def run_detection(self, image):
        with self.detection_graph.as_default():
            with tf.Session(graph=self.detection_graph) as sess:
                image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
                detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
                detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
                detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
                num_detections = detection_graph.get_tensor_by_name('num_detections:0')
                (boxes, scores, classes, num) = sess.run(
                    [detection_boxes, detection_scores, detection_classes, num_detections],
                    feed_dict={image_tensor: image})
                return (boxes, scores, classes, num)

    def judge_traffic_light(scores, classes, num):
        high_score = 0
        class_name = None
        for i in range(num):
            print(classes[i])
            print(scores[i])
            if (scores[i] > high_score):
                high_score = scores[i]
                class_name = classes[i]

        if (class_name != None && high_score > 50):
             return convert_class(class_name)
         else:
             return TrafficLight.UNKNOWN

    def convert_class(self, class_name):
        return class_name #TODO

    def test(self):
        

    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        #TODO implement light color prediction

        return TrafficLight.UNKNOWN

    def save_training_data(self, cv_image, label):
        path = DIR_DATA + FILE_PREFIX_IMG + "{0:%Y%m%d_%H%M%S}_{1}.png".format(datetime.datetime.now(), label)
        cv2.imwrite(path, cv_image)

    def generate_model(self):
        pass


if __name__ == '__main__':
    light_classifier = TLClassifier()
    light_classifier.generate_model()
