from mmdet.apis import init_detector, inference_detector
import numpy as np
from tqdm import tqdm
from glob2 import glob
from utils import saveObjDetect,getAnn,fov2Equir,calIOU,isRightObj,calFailNum,nms
import argparse
import time


def startTest(img_type,model,score_thr,log_file):
    pass_num = 0
    no_pass = 0
    img_content = {}
    IOUs = []
    has_obj = []
    no_obj = []
    fail_obj = 0
    path = '/workspace/dataset/FOV/test/'+img_type
    print(img_type+' total image: ',len(glob(path+'/*.jpg')))
    for img_path in tqdm(glob(path+'/*.jpg')):
        center_pt =img_path[39:-4]
        center_pt = np.array([round(float(center_pt.split('_')[1]),2),round(float(center_pt.split('_')[2]),2)])
        img_name = img_path[34:39]  # get image name
        result = inference_detector(model,img_path)  # get results

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
        if bboxes.shape[0] == 0:
            continue
        labels = labels[inds]  # get labels greater than threshold
        labels = labels.reshape((bboxes.shape[0],1))
        dets = np.hstack((bboxes,labels))
        num = fov2Equir(dets,center_pt,img_name)
        pass_num += num[0]
        no_pass += num[1]
        saveObjDetect(img_name,dets,img_content)

    testset = []
    with open('/workspace/dataset/Pano-dataset/ImageSets/Main/test.txt') as f:  # get the images' names
        for line in f:
            testset.append(line[:-1])
    # testset = testset[0:100]

    for img_name in testset:
        if img_name in img_content:
            has_obj.append(img_name)
        else:
            no_obj.append(img_name)
    print('Detection finished')
    print(len(has_obj),' imgs have been detected with objs')

    print(calFailNum(no_obj),' imgs failed to find obj')

    print('Implement NMS')
    
    for img_name in tqdm(has_obj):
        dets = img_content[img_name]
        picked = nms(dets)
        img_content[img_name] = picked

    print('Compare picked bbox with ground truth')


    for img_name in tqdm(has_obj):
        img_IOUs = []  # to store best IOU for each img
        ground_truths = getAnn(img_name)  # get all the gt for this img
        for gt in ground_truths:  # for each gt find the best bbox
            if isRightObj(gt['label']):
                continue
            iou_max = 0
            index = 0
            for i,element in enumerate(img_content[img_name]):
                iou = calIOU(element,gt['bbox'])
                if iou > iou_max:
                    iou_max = iou
                    index = i
            if iou_max > 0:
                img_IOUs.append(iou_max)  # if max IOU is larger than 0,then add it to the img IOU
                img_content[img_name] = np.delete(img_content[img_name],index,axis=0)  # del this bbox
            if iou_max == 0:
                fail_obj += 1
        
        for bbox in img_content[img_name]:  # for each bbox find best gt
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
                bad_img = open('/workspace/mmdetection-master/log/bad_'+log_file+'.txt','a')
                bad_img.write('\n'+img_name+':'+str(img_average))
                bad_img.close()
            IOUs.append(img_average)

    print(img_type)
    print('\nAverage IOU: ',np.mean(IOUs))
    print('Image which find object: ', len(has_obj)/len(testset))
    print(fail_obj,' objs fail')
    iou_result = open('/workspace/mmdetection-master/log/'+'IOU_'+log_file+'.txt','a')
    iou_result.write(img_type+'\n')
    iou_result.write('\nAverage IOU: '+str(np.mean(IOUs)))
    iou_result.write('\nImage which find object: '+str(len(has_obj)/len(testset)))
    iou_result.write('\nObjs failed: '+str(fail_obj))
    iou_result.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model',dest='model',choices=['faster-rcnn','fast-rcnn','cascade-rcnn','ssd512'],
                        default='cascade-rcnn',help='choose model to use',type=str)
    parser.add_argument('--FOV',dest='FOV',choices=['test','normal'],default='normal',type=str)
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
    if args.FOV == 'test':
        split_type = ['type0']
    else:
        split_type = ['type1','type2','type3','type4']
    
    start = time.time()
    for split in split_type:
        startTest(split,model,args.threshold,args.model)
    print(args.model)
    end = time.time()
    print('time:',end-start)
