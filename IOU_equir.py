from mmdet.apis import init_detector, inference_detector
import numpy as np
from tqdm import tqdm
from glob2 import glob
from utils_equir import saveObjDetect,getAnn,fov2Equir,calIOU,isRightObj,calFailNum,nms
import argparse
import time

config_file = 'configs/ssd512_pano.py'
score_thr = 0.5



path = '/workspace/dataset/Pano-dataset/JPEGImages/'
testset = []
with open('/workspace/dataset/Pano-dataset/ImageSets/Main/test.txt') as f:  # get the images' names
    for line in f:
        testset.append(line[:-1])
# testset = testset[1:201]
checkpoints = [i for i in np.linspace(1,30,30,dtype=int)]
result = []

def startTest(model,model_name):
    global result
    gt_pass = 0
    IOUs = []
    no_objs = []
    fail_obj = 0
    # checkpoint_file = '/workspace/mmdetection-master/checkpoints/my_checkpoints/workdir/epoch_'+str(checkpoint_num)+'.pth'
    # model = init_detector(config_file, checkpoint_file, device='cuda:0')
    print('Start to evaluate checkpoint ',model_name)

    for img_name in tqdm(testset):
        img_path = path+img_name+'.jpg'
        result = inference_detector(model,img_path)
        if isinstance(result, tuple):  # do segmentation or not
            bbox_result, segm_result = result
        else:
            bbox_result, segm_result = result, None
        
        bboxes = np.vstack(bbox_result)
        labels = [
                np.full(bbox.shape[0], i, dtype=np.int32)
                for i, bbox in enumerate(bbox_result)
            ]
        labels = np.concatenate(labels)  # get labels' index
        scores = bboxes[:, -1]  # get scores for each obj detected
        inds = scores > score_thr  # select the score greater than threshold
        bboxes = bboxes[inds, :]
        labels = labels[inds]  # get labels greater than threshold
        if bboxes.shape[0] == 0:
            no_objs.append(img_name)
            continue
        labels = labels.reshape((bboxes.shape[0],1))
        dets = np.hstack((bboxes,labels))

        img_IOUs = []
        ground_truths = getAnn(img_name)
        for gt in ground_truths:
            if isRightObj(gt['label']):
                continue
            iou_max = 0
            index = 0
            for i,element in enumerate(dets):
                iou = calIOU(element,gt['bbox'])
                if iou > iou_max:
                    iou_max = iou
                    index = i
            if iou_max > 0:
                img_IOUs.append(iou_max)
                dets = np.delete(dets,index,axis=0)
            elif iou_max == 0:
                fail_obj += 1
        
        for bbox in dets:  # for each bbox find best gt
            iou_max = 0
            for gt in ground_truths:
                if isRightObj(gt['label']):
                    continue
                iou = calIOU(bbox,gt['bbox'])
                if iou > iou_max:
                    iou_max = iou
            img_IOUs.append(iou_max)
        if len(img_IOUs) > 0:
            img_average = np.mean(img_IOUs)
            if img_average <= 0.2:
                bad_img = open('/workspace/mmdetection-master/log/bad_equir'+model_name+'.txt','a')
                bad_img.write('\n'+img_name+':'+str(img_average))
                bad_img.close()
            IOUs.append(img_average)

    iou = np.mean(IOUs)
    result.append(iou)
    print('checkpoint ',model_name)
    print('Average IOU: ',iou)
    print(calFailNum(no_objs),' imgs failed to find obj')
    print(fail_obj,' objs fail')
    iou_result = open('/workspace/mmdetection-master/log/IOU_equir'+model_name+'.txt','a')
    iou_result.write('\n'+str(model_name)+'\n')
    iou_result.write('\nAverage IOU: '+str(iou))
    iou_result.write('\nImage failed to find: '+str(len(no_objs)/len(testset)))
    iou_result.write('\nObjs failed: '+str(fail_obj))
    iou_result.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model',dest='model',choices=['faster-rcnn','fast-rcnn','cascade-rcnn','ssd512'],
                        default='cascade-rcnn',help='choose model to use',type=str)
    parser.add_argument('--score_threshold',dest='threshold',default=0.5,type=float)
    args = parser.parse_args()

    if args.model == 'faster-rcnn':
        config_file = 'configs/faster_rcnn_x101_64x4d_fpn_1x.py'
        checkpoint_file = '/workspace/mmdetection-master/checkpoints/faster_rcnn_x101_64x4d_fpn_1x_20181218-c9c69c8f.pth'
    elif args.model == 'fast-rcnn':
        config_file = 'configs/faster_rcnn_r50_fpn_1x.py'
        checkpoint_file = '/workspace/mmdetection-master/checkpoints/fast_rcnn_r50_fpn_1x_20181010-08160859.pth'
    elif args.model == 'cascade-rcnn':
        config_file = 'configs/cascade_rcnn_x101_64x4d_fpn_1x.py'
        checkpoint_file = '/workspace/mmdetection-master/checkpoints/cascade_rcnn_x101_64x4d_fpn_1x_20181218-e2dc376a.pth'
    elif args.model == 'ssd512':
        config_file = 'configs/ssd512_coco.py'
        checkpoint_file = '/workspace/mmdetection-master/checkpoints/ssd512_coco_vgg16_caffe_120e_20181221-d48b0be8.pth'
    
    model = init_detector(config_file, checkpoint_file, device='cuda:0')
    
    start = time.time()
    for checkpoint in checkpoints:
        startTest(model,args.model)
    print(args.model)
    end = time.time()
    print('time:',end-start)

    # print('Total average: ',np.mean(result))