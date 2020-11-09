from mmdet.apis import init_detector, inference_detector
import mmcv
from pyson.vision import show
import numpy as np 

def _bbox_threshold(result, score_thr=0.3):
    """
        input: 
            result:predicted tensor of [None, 5] of a detector
        Return bboxes, labels
        Examples:
            result = inference_detector(model, data)
            bboxes, labels = bbox_threshold(result, score_thr=0.25)
    """
    bbox_result = np.array(result)
    bboxes = np.vstack(bbox_result)
    labels = [
        np.full(bbox.shape[0], i, dtype=np.int32)
        for i, bbox in enumerate(bbox_result)
    ]
    labels = np.concatenate(labels)
    scores = bboxes[:, -1]
    inds = scores > score_thr
    bboxes = bboxes[inds, :]
    labels = labels[inds]
    return bboxes, labels


class DetModel():
    def __init__(self, config_path, checkpoint_path):
        self.model = init_detector(config_path, checkpoint_path)

    def pred(self, img):
        result = inference_detector(self.model, img)
        return result

    def bbox_threshold(self, result, score_thr=0.3):
        return _bbox_threshold(result, score_thr)

    def draw_image(self, path, result):
        img_color = self.model.show_result(path, result)
        return img_color


if __name__ == '__main__':
    config_path = '/home/haianh/gitprojects/bbdetection/configs/za_challenge/vfnet_r50_za_v2.py'
    checkpoint_path = '/home/haianh/gitprojects/bbdetection/work_dirs/vfnet_r50_za_v2_full_ens/epoch_12.pth'
    det_model = DetModel(config_path, checkpoint_path)
    from glob import glob
    
    paths = glob('/home/haianh/gitprojects/bbdetection/data/za_traffic_2020/traffic_public_test/images/*.png')
    path = paths[0]
    result = det_model.pred(path)

    img = det_model.draw_image(path, result)
    # mmcv.imwrite('test.jpg')
    show(img)