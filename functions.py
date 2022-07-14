import numpy as np
import cv2
import json
import urllib

def url_to_image(url):
    resp = urllib.request.urlopen(url)
    image = np.asarray(bytearray(resp.read()), dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    return image

def get_center_point(coordinate_dict):
    points = dict()
    for key in coordinate_dict.keys():
        xmin, ymin, xmax, ymax = coordinate_dict[key][0]
        x_center = (xmin + xmax) / 2
        y_center = (ymin + ymax) / 2
        points[key] = (x_center, y_center)
    return points

def find_miss_corner(coordinate_dict):
    dict_corner = ['top_left', 'top_right', 'bottom_left', 'bottom_right']
    for key in dict_corner:
        if key not in coordinate_dict.keys():
            return key

def calculate_missed_coord_corner(coordinate_dict):
    #calulate a coord corner of a rectangle 
    def calculate_coord_by_mid_point(coor1, coord2, coord3):
        midpoint = np.add(coordinate_dict[coor1], coordinate_dict[coord2]) / 2
        y = 2 * midpoint[1] - coordinate_dict[coord3][1]
        x = 2 * midpoint[0] - coordinate_dict[coord3][0]
        return (x, y)
    # calculate missed corner coordinate
    corner = find_miss_corner(coordinate_dict)
    if corner == 'top_left':
        coordinate_dict['top_left'] = calculate_coord_by_mid_point('top_right', 
        'bottom_left', 'bottom_right')
    elif corner == 'top_right':
        coordinate_dict['top_right'] = calculate_coord_by_mid_point('top_left', 
        'bottom_right', 'bottom_left')
    elif corner == 'bottom_left':
        coordinate_dict['bottom_left'] = calculate_coord_by_mid_point('top_left', 
        'bottom_right', 'top_right')
    elif corner == 'bottom_right':
        coordinate_dict['bottom_right'] = calculate_coord_by_mid_point('bottom_left', 
        'top_right', 'top_left')
    return coordinate_dict

def perspective_transform(image, source_points):
    dest_points = np.float32([[0, 0], [500, 0], [500, 300], [0, 300]])
    M = cv2.getPerspectiveTransform(source_points, dest_points)
    dst = cv2.warpPerspective(image, M, (500, 300))
    return dst

def align_image(image, coordinate_dict):
    if len(coordinate_dict) < 3:
        raise ValueError('Please try again')
    # convert (xmin, ymin, xmax, ymax) to (x_center, y_center)
    coordinate_dict = get_center_point(coordinate_dict)
    if len(coordinate_dict) == 3:
        coordinate_dict = calculate_missed_coord_corner(coordinate_dict)
    top_left_point = coordinate_dict['top_left']
    top_right_point = coordinate_dict['top_right']
    bottom_right_point = coordinate_dict['bottom_right']
    bottom_left_point = coordinate_dict['bottom_left']
    source_points = np.float32([top_left_point, top_right_point, bottom_right_point, bottom_left_point])
    # transform image and crop
    crop = perspective_transform(image, source_points)
    return crop

def load_labels_map(label_url):
    # Opening JSON file
    f = open(label_url)
    data = json.load(f)
    if data:
        return data['labels']
    return []

def process_output(type, data, threshold, targetSize):
    scores, boxes, classes = None, None, None
    label_map = None

    if type == 'corner':
        scores = data['detection_scores']
        boxes = data['detection_boxes']
        classes = data['detection_classes']
        label_map = load_labels_map('label_map/corner.pbtxt')
    
    if type == 'text':
        scores = data['detection_scores']
        boxes = data['detection_boxes']
        classes = data['detection_classes']
        label_map = load_labels_map('label_map/text.pbtxt')

    results = {}

    for i in range(len(scores)):
        if scores[i] > threshold:
            label = label_map[int(classes[i]) - 1]
            if label in results:
                results[label].append([
                    boxes[i][1] * targetSize['w'], 
                    boxes[i][0] * targetSize['h'], 
                    boxes[i][3] * targetSize['w'], 
                    boxes[i][2] * targetSize['h']])
            else:
                results[label] = [[
                    boxes[i][1] * targetSize['w'], 
                    boxes[i][0] * targetSize['h'], 
                    boxes[i][3] * targetSize['w'], 
                    boxes[i][2] * targetSize['h']]]
    
    return results