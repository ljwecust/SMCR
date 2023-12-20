import torch
import numpy as np
import copy
from PIL import ImageDraw, Image

def sortProbsRes(Probs_list):
    Probs = torch.FloatTensor(Probs_list)
    _, sorted_idx = Probs.topk(10, largest=True, sorted=True)
    return sorted_idx.tolist()


class CropImg(object):
    def __init__(self, image):
        self.image = image
    
    def _xywh2lurl(self, xywh_list):
        x = xywh_list[0]
        y = xywh_list[1]
        w = xywh_list[2]
        h = xywh_list[3]
        return (x, y, x + w, y + h)

    def crop_lurl(self, bbox_lurl):
        return self.image.crop(bbox_lurl)

    def crop_xywh(self, bbox_xywh):
        return self.crop_lurl(self._xywh2lurl(bbox_xywh))
    
    def showbbox_lurl(self, *bbox_lurl):
        color = [(0,0,0),(255,255,0),(25,25,112),(255,0,0),(0,0,255)]
        width=1
        image = copy.deepcopy(self.image)
        draw = ImageDraw.Draw(image)

        i = 0
        for bbox in bbox_lurl:
            draw.rectangle(bbox, outline=color[i%len(color)], width=width)
            i += 1
        return image
    
    def showbbox_xywh(self, *bbox_xywh):
        bbox_lurl = [self._xywh2lurl(bbox) for bbox in bbox_xywh]
        return self.showbbox_lurl(*bbox_lurl)

