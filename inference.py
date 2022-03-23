#필요한 모듈
import sys
sys.path.append('/home/wmit/r-cnn_service')

import os
import math
import numpy as np
from tqdm import tqdm
from skimage.io import imread
from google.protobuf import text_format
import json
import cv2
import copy
import logging
import time
import visualization_utils as vis_utils
import csv
from builders import model_builder
from protos import pipeline_pb2
from utils.np_rbox_ops import non_max_suppression
import six
import glob
import random
from pdf_reports import pug_to_html, write_report
import label_map_util
#import medbiz
#from hdfs import InsecureClient
#from hdfs3 import HDFileSystem
from PIL import Image
import tensorflow as tf
slim = tf.contrib.slim

#GPU설정

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="2,3,4"


def get_detection_graph(pipeline_config_path):
    """pipline_config_path에서 그래프 작성
    :param: str pipeline_config_path : 파이프 라인 구성 파일의 경로
    :return: 그래프
    """
    pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
    with tf.io.gfile.GFile(pipeline_config_path, 'r') as f:
        text_format.Merge(f.read(), pipeline_config)

    detection_model = model_builder.build(pipeline_config.model, is_training=False)
    input_tensor = tf.compat.v1.placeholder(dtype=tf.uint8, shape=(None, None, None, 3), name='image_tensor')
    inputs = tf.cast(input_tensor,'float32')
    preprocessed_inputs = detection_model.preprocess(inputs)
    output_tensors = detection_model.predict(preprocessed_inputs)
    postprocessed_tensors = detection_model.postprocess(output_tensors)

    output_collection_name = 'inference_op'
    boxes = postprocessed_tensors.get('detection_boxes')
    scores = postprocessed_tensors.get('detection_scores')
    classes = postprocessed_tensors.get('detection_classes') + 1
    num_detections = postprocessed_tensors.get('num_detections')
    outputs = dict()
    outputs['detection_boxes'] = tf.identity(boxes, name='detection_boxes')
    outputs['detection_scores'] = tf.identity(scores, name='detection_scores')
    outputs['detection_classes'] = tf.identity(classes, name='detection_classes')
    outputs['num_detections'] = tf.identity(num_detections, name='num_detections')
    for output_key in outputs:
        tf.compat.v1.add_to_collection(output_collection_name, outputs[output_key])

    graph = tf.compat.v1.get_default_graph()
    return graph

def get_patch_generator(image, patch_size, overlay_size):
    """ 그리드로 이미지를 분할하는 패치 생성기
    :param numpy image: 원래 이미지
    :param int patch_size: 패치의 너비와 높이가 같은 패치 크기
    :param overlay_size: 패치의 오버레이 크기
    :return: 패치 이미지, 행 및 열 좌표 생성기
    """
    step = patch_size - overlay_size
    for row in range(0, image.shape[0] - overlay_size, step):
        for col in range(0, image.shape[1] - overlay_size, step):
            # Handling for out of bounds
            patch_image_height = patch_size if image.shape[0] - row > patch_size else image.shape[0] - row
            patch_image_width = patch_size if image.shape[1] - col > patch_size else image.shape[1] - col

            # Set patch image
            patch_image = image[row: row + patch_image_height, col: col + patch_image_width]

            # Zero padding if patch image is smaller than patch size
            if patch_image_height < patch_size or patch_image_width < patch_size:
                pad_height = patch_size - patch_image_height
                pad_width = patch_size - patch_image_width
                patch_image = np.pad(patch_image, ((0, pad_height), (0, pad_width), (0, 0)), 'constant')

            yield patch_image, row, col


def load_geojson(filename):
    """ geojson 레이블 파일에서 레이블 데이터를 가져옵니다.
    :param (str) filename: geojson 레이블 파일의 파일 경로
    :return: (numpy.ndarray, numpy.ndarray, numpy.ndarray) 각 진리에 대한 좌표, 이미지 이름 및 클래스 코드에 해당하는 좌표, 칩 및 클래스입니다.
    """
    with open(filename) as f:
        data = json.load(f)

    obj_coords = np.zeros((len(data['features']), 8))
    image_ids = np.zeros((len(data['features'])), dtype='object')
    class_indices = np.zeros((len(data['features'])), dtype=int)
    class_names = np.zeros((len(data['features'])), dtype='object')

    for idx in range(len(data['features'])):
        properties = data['features'][idx]['properties']
        image_ids[idx] = properties['image_id']
        obj_coords[idx] = np.array([float(num) for num in properties['bounds_imcoords'].split(",")])
        class_indices[idx] = properties['type_id']
        class_names[idx] = properties['type_name']

    return image_ids, obj_coords, class_indices, class_names

def cvt_coords_to_rboxes(coords):
    """ geojson에서 (cy, cx, height, width, theta) 형식으로 좌표 배열을 처리합니다.

    :param (numpy.ndarray) 좌표 : 4 개의 모서리 점 상자가있는 모양의 배열 (N, 8)
    :return: (numpy.ndarray) 적절한 형식의 좌표를 가진 모양 (N, 5)의 배열
    """
    rboxes = []
    for coord in coords:
        pts = np.reshape(coord, (-1, 2)).astype(dtype=np.float32)
        (cx, cy), (width, height), theta = cv2.minAreaRect(pts)

        if width < height:
            width, height = height, width
            theta += 90.0
        rboxes.append([cy, cx, height, width, math.radians(theta)])

    return np.array(rboxes)
def visualize_detection_results(result_dict,categories,agnostic_mode=False,show_groundtruth=True,min_score_thresh=0.5,max_num_predictions=None):
    """탐지 결과를 시각화하고 시각화를 이미지(png)파일로 생성

    이 기능은 감지 된 경계 상자로 이미지를 시각화하고 tensorboard에서 볼 수있는 이미지 요약에 씁니다. 
	이미지를 디렉토리에 씁니다. 
	레이블 맵에 항목이 누락 된 경우 시각화에서 알 수없는 클래스 이름이 "N / A"로 표시됩니다.

    Args:
      result_dict: 평가되는 각각의 이미지에 대응하는 진상 및 검출 데이터를 보유하는 사전.  
	  사전에는 다음과 같은 키가 필요합니다.:
          'original_image': 모양 [1, 높이, 너비, 3]의 이미지를 나타내는 numpy 배열
          'detection_boxes': a numpy array of shape [N, 4]
          'detection_scores': a numpy array of shape [N]
          'detection_classes': a numpy array of shape [N]
          'groundtruth_boxes': a numpy array of shape [N, 4]
          'groundtruth_keypoints': a numpy array of shape [N, num_keypoints, 2]
        탐지는 점수의 내림차순 및 표시를 위해 제공되는 것으로 가정되며 점수는 0과 1 사이의 확률이라고 가정합니다.
      categories: 가능한 모든 범주를 나타내는 사전 목록
        Each dict in this list has the following keys:
            'id': (required) an integer id uniquely identifying this category
            'name': (required) string representing category name
              e.g., 'cat', 'dog', 'pizza'
            'supercategory': (optional) string representing the supercategory
              e.g., 'animal', 'vehicle', 'food', etc
      export_dir: 이미지가 작성되는 출력 디렉토리 지정하여 (기본값) 이미지가 내보냅니다.
      agnostic_mode: 클래스 불가 지 모드로 평가할지 여부를 제어하는 부울 (기본값 : False)
      show_groundtruth: boolean (기본값 : True) 감지 된 상자 외에 기본 진리 상자를 표시할지 여부를 제어
      min_score_thresh: 상자를 시각화하기위한 최소 점수 임계 값
      max_num_predictions: 시각화 할 최대 탐지 수
    Raises:
      ValueError : result_dict에 예상 키가 포함되어 있지 않은 경우 에러발생 (예상 키:'original_image', 'detection_boxes', 'detection_scores', 'detection_classes')
    """
    if not set([
        'original_image', 'detection_boxes', 'detection_scores',
        'detection_classes'
    ]).issubset(set(result_dict.keys())):
        raise ValueError('result_dict does not contain all expected keys.')
    if show_groundtruth and ('groundtruth_boxes' not in result_dict and 'groundtruth_rboxes' not in result_dict):
        raise ValueError('If show_groundtruth is enabled, result_dict must contain groundtruth_boxes.')
    logging.info('Creating detection visualizations.')
    category_index = label_map_util.create_category_index(categories)
    image = result_dict['original_image'].copy()
    detection_boxes = result_dict['detection_boxes']
    detection_scores = result_dict['detection_scores']
    detection_classes = np.int32((result_dict['detection_classes']))
    detection_keypoints = result_dict.get('detection_keypoints', None)
    detection_masks = result_dict.get('detection_masks', None)
	#실제 객체를 이미지에 나타냄
    if show_groundtruth:
        groundtruth_boxes = result_dict['groundtruth_boxes']
        groundtruth_keypoints = result_dict.get('groundtruth_keypoints', None)
        vis_utils.visualize_boxes_and_labels_on_image_array(
            image,
            groundtruth_boxes,
            None,
            None,
            category_index,
            keypoints=groundtruth_keypoints,
            use_normalized_coordinates=False,
            max_boxes_to_draw=None)
	# 예측한 객체를 이미지에 나타냄
    vis_utils.visualize_boxes_and_labels_on_image_array(
        image,
        detection_boxes,
        detection_classes,
        detection_scores,
        category_index,
        instance_masks=detection_masks,
        keypoints=detection_keypoints,
        use_normalized_coordinates=False,
        max_boxes_to_draw=max_num_predictions,
        min_score_thresh=min_score_thresh,
        agnostic_mode=agnostic_mode)
    export_path = os.path.join('/home/wmit/r-cnn_service/export.png')
    # 이미지 저장
    vis_utils.save_image_array_as_png(image,export_path)

def inference():
    image_dir   = '/home/wmit/r-cnn_service/data'
    image_path =sys.argv[1]
    pdfFullPath = sys.argv[2]
    pugFullPath ='/home/wmit/r-cnn_service/Satellite_vessel_Detection_20201216.pug'
    pipeline_config_path ='/home/wmit/r-cnn_service/configs/rbox_cnn_resnet101.config'
    ckpt_path ='/home/wmit/r-cnn_service/train/model.ckpt-120000'
    patch_size=1024
    overlay_size=384
    json_path='/home/wmit/r-cnn_service/configs/labels.json'
    image_input = cv2.imread(image_path)
    cv2.imwrite('/home/wmit/r-cnn_service/input.png',image_input)
    # Get filenames
    file_paths = [os.path.join(root, name) for root, dirs, files in os.walk(image_dir) for name in files if name.endswith('png')]
    print(file_paths)
    file_paths =[image_path]
    # Create graph
    graph = get_detection_graph(pipeline_config_path)
    # Inference
    with tf.compat.v1.Session(graph=graph) as sess:
        # 체크 포인트 파일에서 가중치로드
        variables_to_restore = tf.compat.v1.global_variables()
        saver = tf.compat.v1.train.Saver(variables_to_restore)
        saver.restore(sess, ckpt_path)
        # 탐지 모델의 텐서 가져 오기
        image_tensor = graph.get_tensor_by_name('image_tensor:0')
        detection_boxes = graph.get_tensor_by_name('detection_boxes:0')
        detection_scores = graph.get_tensor_by_name('detection_scores:0')
        detection_classes = graph.get_tensor_by_name('detection_classes:0')
        # detection 결과를 저장할 딕셔너리 생성
        det_by_file = dict()
        # grountruth 추가
        image_id, obj_coords, class_indices, class_names = load_geojson(json_path)
        obj_coords = cvt_coords_to_rboxes(obj_coords)
        num =[]
        for file_path in tqdm(file_paths):
            a=file_path.split('/')
            for i in range(len(image_id)):
                ground_truth =[]
                if image_id[i] == a[-1]:
                    ground_truth.append(obj_coords[i])
                    num.append(ground_truth)
            
            image = imread(image_path)
            patch_generator = get_patch_generator(image, patch_size=patch_size, overlay_size=overlay_size)
            classes_list, scores_list, rboxes_list = list(), list(), list()
            for patch_image, row, col in patch_generator:
                classes, scores, rboxes = sess.run([detection_classes, detection_scores, detection_boxes],feed_dict={image_tensor: [patch_image]})
                rboxes = rboxes[0]
                classes = classes[0]
                scores = scores[0]
                rboxes *= [patch_image.shape[0], patch_image.shape[1], patch_image.shape[0], patch_image.shape[1], 1]
                rboxes[:, 0] = rboxes[:, 0] + row
                rboxes[:, 1] = rboxes[:, 1] + col
                rboxes_list.append(rboxes)
                classes_list.append(classes)
                scores_list.append(scores)
            rboxes = np.array(rboxes_list).reshape(-1, 5)
            groundtruth_boxes = np.array(num).reshape(-1, 5)
            classes = np.array(classes_list).flatten()
            scores = np.array(scores_list).flatten()
            rboxes = rboxes[scores > 0]
            classes = classes[scores > 0]
            scores = scores[scores > 0]
            indices = non_max_suppression(rboxes, scores, iou_threshold=0.3)
            rboxes = rboxes[indices]
            classes = classes[indices]
            scores = scores[indices]
            det_by_file[file_path]={'original_image': image,'detection_boxes': rboxes, 'detection_scores': scores,'detection_classes': classes,'groundtruth_boxes':groundtruth_boxes}
            label_map=label_map_util.load_labelmap('/home/wmit/r-cnn_service/configs/label_map.pbtxt')
            categories=label_map_util.convert_label_map_to_categories(label_map,50,use_display_name=True)
            visualize_detection_results(det_by_file[file_path],categories=categories)
            html = pug_to_html(pugFullPath, title="report")
            write_report(html,pdfFullPath)
if __name__ == '__main__':
    inference()
