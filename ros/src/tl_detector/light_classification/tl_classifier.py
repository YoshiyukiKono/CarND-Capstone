from styx_msgs.msg import TrafficLight
import datetime
import cv2
import sys
import os

import tensorflow as tf
import numpy as np
from PIL import Image

import yaml

PATH_TF_MODELS_RESEARCH = "/home/student/github/models/research"
PATH_TF_MODELS_SLIM = "/home/student/github/models/research/slim"
PATH_TF_MODELS_OBJECT_DETECTION = "/home/student/github/models/research/object_detection"

sys.path.append(PATH_TF_MODELS_RESEARCH)
sys.path.append(PATH_TF_MODELS_SLIM)
sys.path.append(PATH_TF_MODELS_OBJECT_DETECTION)
from utils import label_map_util
from utils import visualization_utils as vis_util

from utils import dataset_util

FILE_PREFIX_IMG = "IMG_"
DIR_DATA = "DATA/"

PATH_TEST_IMAGE_FILE = "/home/student/CarND-Capstone/ros/src/tl_detector/DATA/IMG_20180701_104354_0.png"

PATH_TRAINED_GRAPH = "/home/student/github/models/research/object_detection/ssd_mobilenet_v1_coco_2017_11_17/frozen_inference_graph.pb"
PATH_LABEL_MAP = "/home/student/github/models/research/object_detection/data/mscoco_label_map.pbtxt"
NUM_CLASSES = 90

PATH_TRAIN_DATA_DIR = ""
PATH_YAML = "/home/student/Downloads/train.yaml" #TODO change the path
WIDTH_TRAIN_DATA = 1280
HEIGHT_TRAIN_DATA = 720
NUM_CLASSES_TRAIN_DATA = 14
PATH_TF_RECORD = "train_data.tfrecords"
DICT_LABEL = { "Green" : 1, "Red" : 2, "GreenLeft" : 3, "GreenRight" : 4,
    "RedLeft" : 5, "RedRight" : 6, "Yellow" : 7, "off" : 8,
    "RedStraight" : 9, "GreenStraight" : 10, "GreenStraightLeft" : 11, "GreenStraightRight" : 12,
    "RedStraightLeft" : 13, "RedStraightRight" : 14 }

class TLClassifier(object):
    def __init__(self):
        #TODO load classifier
        #self.append_required_modules()
        self.load_model()
        self.load_label_map()
        #pass

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
                image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
                detection_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
                detection_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
                detection_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
                num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')
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

        if (class_name != None & high_score > 50):
             return convert_class(class_name)
        else:
             return TrafficLight.UNKNOWN

    def convert_class(self, class_name):
        return class_name #TODO

    def test(self):
        image = Image.open(PATH_TEST_IMAGE_FILE)
        (width, height) = image.size
        image_array = np.array(image.getdata()).reshape((height, width, 3)).astype(np.uint8)
        image_one_h_w_c = np.expand_dims(image_array, axis=0)
        (boxes, scores, classes, num) = self.run_detection(image_one_h_w_c)
        vis_util.visualize_boxes_and_labels_on_image_array(image_array,
            np.squeeze(boxes), np.squeeze(classes).astype(np.int32), np.squeeze(scores),
            self.category_index, use_normalized_coordinates=True, line_thickness=8)
        #image_np
        cv2.imwrite("test.png", image_array)

    def write_tf_record(self):
        writer = tf.python_io.TFRecordWriter(PATH_TF_RECORD)
        examples = yaml.load(open(PATH_YAML, 'rb').read())
        for example in examples:
            filename = example['path']

            if (not os.path.exists(filename)):
                print(filename, " does not exist.")
                continue
            filename = filename.encode()
            with tf.gfile.GFile(filename, 'rb') as fid:
                encoded_image = fid.read()
            image_format = 'png'.encode()
            xmins = []
            xmaxs = []
            ymins = []
            ymaxs = []
            classes_text = []
            classes = []
            for box in example['boxes']:
                xmins.append(float(box['x_min']/width))
                xmaxs.append(float(box['x_max']/width))
                ymins.append(float(box['y_min']/height))
                ymaxs.append(float(box['y_max']/height))
                classes_text.append(box['label'])
                classes.append(int(DICT_LABEL[box['label']]))
            tf_example = tf.train.Example(features=tf.train.Features(feature={
                'image/height' : dataset_util.int64_feature(height),
                'image/width' : dataset_util.int64_feature(width),
                'image/filename' : dataset_util.bytes_feature(filename),
                'image/source_id' : dataset_util.bytes_feature(filename),
                'image/encoded' : dataset_util.bytes_feature(encoded_image),
                'image/format' : dataset_util.bytes_feature(image_format),
                'image/object/bbox/xmin' : dataset_util.float_list_feature(xmins),
                'image/object/bbox/xmax' : dataset_util.float_list_feature(xmaxs),
                'image/object/bbox/ymin' : dataset_util.float_list_feature(ymins),
                'image/object/bbox/ymax' : dataset_util.float_list_feature(ymaxs),
                'image/object/class/text' : dataset_util.bytes_list_feature(classes_text),
                'image/object/class/label' : dataset_util.int64_list_feature(classes),
            }))
            writer.write(tf_example.SerializeToString())
        writer.close()

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
    light_classifier.test() # generate_model()
    if (not os.path.exists(PATH_TF_RECORD)):
        light_classifier.write_tf_record()
