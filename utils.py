import mmcv
import numpy as np
import xml.etree.cElementTree as ET
from math import pi



# labels = ['person', 'bicycle', 'car', 'motorcycle', 'bus', 'truck', 'boat', 'bird', 'cat', 'dog', 'elephant', 'skateboard', 'surfboard']
labels = ['person', 'car','bus']
# labels = [0, 1, 2, 3, 5, 7, 8, 14, 15, 16, 20, 36, 37]


class NFOV():
    def __init__(self, height=400, width=800):
        self.FOV = [0.45, 0.45]
        self.PI = pi
        self.PI_2 = pi * 0.5
        self.PI2 = pi * 2.0
        self.height = height
        self.width = width
        self.screen_points = self._get_screen_img()

    def _get_coord_rad(self, isCenterPt, center_point=None):
        return (center_point * 2 - 1) * np.array([self.PI, self.PI_2]) \
            if isCenterPt \
            else \
            (self.screen_points * 2 - 1) * np.array([self.PI, self.PI_2]) * (
                np.ones(self.screen_points.shape) * self.FOV)

    def _get_screen_img(self):
        xx, yy = np.meshgrid(np.linspace(0, 1, self.width), np.linspace(0, 1, self.height))
        return np.array([xx.ravel(), yy.ravel()]).T

    def _calcSphericaltoGnomonic(self, convertedScreenCoord):
        x = convertedScreenCoord.T[0]
        y = convertedScreenCoord.T[1]

        rou = np.sqrt(x ** 2 + y ** 2)
        c = np.arctan(rou)
        sin_c = np.sin(c)
        cos_c = np.cos(c)

        lat = np.arcsin(cos_c * np.sin(self.cp[1]) + (y * sin_c * np.cos(self.cp[1])) / rou)
        lon = self.cp[0] + np.arctan2(x * sin_c, rou * np.cos(self.cp[1]) * cos_c - y * np.sin(self.cp[1]) * sin_c)

        lat = (lat / self.PI_2 + 1.) * 0.5
        lon = (lon / self.PI + 1.) * 0.5

        return np.array([lon, lat]).T

    def _bilinear_interpolation(self, screen_coord):
        uf = np.mod(screen_coord.T[0],1) * self.frame_width  # long - width
        vf = np.mod(screen_coord.T[1],1) * self.frame_height  # lat - height

        x0 = np.floor(uf).astype(int)  # coord of pixel to bottom left
        y0 = np.floor(vf).astype(int)
        _x2 = np.add(x0, np.ones(uf.shape).astype(int))  # coords of pixel to top right
        y2 = np.add(y0, np.ones(vf.shape).astype(int))

        x2 = np.mod(_x2, self.frame_width)
        y2 = np.minimum(y2, self.frame_height - 1)

        base_y0 = np.multiply(y0, self.frame_width)
        base_y2 = np.multiply(y2, self.frame_width)

        A_idx = np.add(base_y0, x0)
        B_idx = np.add(base_y2, x0)
        C_idx = np.add(base_y0, x2)
        D_idx = np.add(base_y2, x2)

        flat_img = np.reshape(self.frame, [-1, self.frame_channel])

        A = np.take(flat_img, A_idx, axis=0)
        B = np.take(flat_img, B_idx, axis=0)
        C = np.take(flat_img, C_idx, axis=0)
        D = np.take(flat_img, D_idx, axis=0)

        wa = np.multiply(_x2 - uf, y2 - vf)
        wb = np.multiply(_x2 - uf, vf - y0)
        wc = np.multiply(uf - x0, y2 - vf)
        wd = np.multiply(uf - x0, vf - y0)

        # interpolate
        AA = np.multiply(A, np.array([wa, wa, wa]).T)
        BB = np.multiply(B, np.array([wb, wb, wb]).T)
        CC = np.multiply(C, np.array([wc, wc, wc]).T)
        DD = np.multiply(D, np.array([wd, wd, wd]).T)
        nfov = np.reshape(np.round(AA + BB + CC + DD).astype(np.uint8), [self.height, self.width, 3])
        return nfov

    def toNFOV(self, frame, center_point):
        self.frame = frame
        self.frame_height = frame.shape[0]
        self.frame_width = frame.shape[1]
        self.frame_channel = frame.shape[2]

        self.cp = self._get_coord_rad(center_point=center_point, isCenterPt=True)
        convertedScreenCoord = self._get_coord_rad(isCenterPt=False)
        spericalCoord = self._calcSphericaltoGnomonic(convertedScreenCoord)
        return spericalCoord

nfov = NFOV()

def saveObjDetect(img_name,dets,img_content):
    if img_name not in img_content: 
        img_content[img_name] = dets
    else:
        img_content[img_name] = np.vstack((img_content[img_name],dets))

def getAnn(img_name):
    img_ann = []
    xml_path = '/workspace/dataset/Pano-dataset/Annotations/'+img_name+'.xml'
    tree = ET.ElementTree(file=xml_path)
    root = tree.getroot()
    ObjectsSet = root.findall('object')
    for Object in ObjectsSet:
        objname = Object.find('name').text
        bndbox = Object.find('bndbox')
        ground_truth = [round(float(bndbox.find('xmin').text)),round(float(bndbox.find('ymin').text)),
                        round(float(bndbox.find('xmax').text)),round(float(bndbox.find('ymax').text))]
        img_ann.append({'label':objname,'bbox':ground_truth})
    return img_ann

def fov2Equir(dets,center_point,img_name):
    pass_num = 0
    no_pass = 0
    equir_img = '/workspace/dataset/Pano-dataset/JPEGImages/'+img_name+'.jpg'
    row  = dets.shape[0]
    col = dets.shape[1]
    for i in range(dets.shape[0]):
        bbox = dets[i,0:4]
        pt = [0,0,0,0]
        for j in range(4):
            pt[j] = bbox[j].astype(np.int32)
        if pt[0] > 400 or pt[2] > 400 or pt[1] > 800 or pt[3] > 800:
            pass_num+=1
            continue
        no_pass += 1
        bbox_equir = bbox
        alpha = 0.4
        equir_frame2 = mmcv.imread(equir_img)
        screen_cords= nfov.toNFOV(equir_frame2, center_point)
        y = np.mod(screen_cords.T[1], 1)
        x = np.mod(screen_cords.T[0], 1)
        
        for k in range(0,3,2):
            pt_FOV = np.array([pt[k+1],pt[k]])
            FoV_cp_ravel = np.ravel_multi_index(pt_FOV, [400, 800])
            pt_in_equir = np.array([x[FoV_cp_ravel], y[FoV_cp_ravel]])
            pix_in_equir = pt_in_equir * np.array([equir_frame2.shape[1], equir_frame2.shape[0]])
            bbox_equir[k],bbox_equir[k+1] = pix_in_equir[0],pix_in_equir[1]
        dets[i,0:4] = np.array(bbox_equir)
    return (pass_num,no_pass)



def calIOU(det_bbox,gt_bbox):
    """
    自定义函数，计算两矩形 IOU，传入为均为矩形对角线，（x,y）  坐标。
    """
    x1 = det_bbox[0]
    y1 = det_bbox[1]
    width1 = det_bbox[2]-det_bbox[0]
    height1 = det_bbox[3]-det_bbox[1]

    x2 = gt_bbox[0]
    y2 = gt_bbox[1]
    width2 = gt_bbox[2]-gt_bbox[0]
    height2 = gt_bbox[3]-gt_bbox[1]

    endx = max(x1+width1,x2+width2)
    startx = min(x1,x2)
    width = width1+width2-(endx-startx)

    endy = max(y1+height1,y2+height2)
    starty = min(y1,y2)
    height = height1+height2-(endy-starty)

    if width <=0 or height <= 0:
        ratio = 0 # 重叠率为 0 
    else:
        Area = width*height # 两矩形相交面积
        Area1 = width1*height1
        Area2 = width2*height2
        ratio = Area*1./(Area1+Area2-Area)
    # return IOU
    return ratio


def isRightObj(gt_label):  # check if the detected obj correspond to the gt label
    if gt_label in labels:
        return True
    else:
        return False


def calFailNum(no_obj):
    det_fail_num = 0
    for img_name in no_obj:
        xml_path = '/workspace/dataset/Pano-dataset/Annotations/'+img_name+'.xml'
        try:
            tree = ET.ElementTree(file=xml_path)
        except FileNotFoundError:
            print(img_name)
            continue
        root = tree.getroot()
        ObjectSet = root.findall('object')
        if len(ObjectSet) > 0:
            det_fail_num += 1
    return det_fail_num

def nms(dets, threshold=0.5):
    # If no bounding boxes, return empty list
    if len(dets) == 0:
        return np.array([])
    # det = [x1,y1,x2,y2,score,label]

    # coordinates of bounding boxes
    start_x = dets[:, 0]
    start_y = dets[:, 1]
    end_x = dets[:, 2]
    end_y = dets[:, 3]

    # Confidence scores of bounding boxes
    score = dets[:, 4]

    # Picked bounding boxes
    picked_boxes = []
    picked_score = []

    # Compute areas of bounding boxes
    areas = (end_x - start_x + 1) * (end_y - start_y + 1)

    # Sort by confidence score of bounding boxes
    order = np.argsort(score)

    # Iterate bounding boxes
    while order.size > 0:
        # The index of largest confidence score
        index = order[-1]

        # Pick the bounding box with largest confidence score
        picked_boxes.append(dets[index])
        picked_score.append(score[index])

        # Compute ordinates of intersection-over-union(IOU)
        x1 = np.maximum(start_x[index], start_x[order[:-1]])
        x2 = np.minimum(end_x[index], end_x[order[:-1]])
        y1 = np.maximum(start_y[index], start_y[order[:-1]])
        y2 = np.minimum(end_y[index], end_y[order[:-1]])

        # Compute areas of intersection-over-union
        w = np.maximum(0.0, x2 - x1 + 1)
        h = np.maximum(0.0, y2 - y1 + 1)
        intersection = w * h

        # Compute the ratio between intersection and union
        ratio = intersection / (areas[index] + areas[order[:-1]] - intersection)

        left = np.where(ratio < threshold)
        order = order[left]

    picked_boxes = np.array(picked_boxes)
    return picked_boxes