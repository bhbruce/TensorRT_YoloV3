import math
from PIL import Image
import numpy as np
import os
import torch


class PostprocessYOLO(object):
    """Class for post-processing the three outputs tensors from YOLOv3-608."""

    def __init__(self,
                 yolo_masks,
                 yolo_anchors,
                 num_classes,
                 stride=[32, 16, 8]):
                #  obj_threshold,
                #  nms_threshold,
                #  yolo_input_resolution):
        """Initialize with all values that will be kept when processing several frames.
        Assuming 3 outputs of the network in the case of (large) YOLOv3.

        Keyword arguments:
        yolo_masks -- a list of 3 three-dimensional tuples for the YOLO masks
        yolo_anchors -- a list of 9 two-dimensional tuples for the YOLO anchors
        object_threshold -- threshold for object coverage, float value between 0 and 1
        nms_threshold -- threshold for non-max suppression algorithm,
        float value between 0 and 1
        input_resolution_yolo -- two-dimensional tuple with the target network's (spatial)
        input resolution in HW order
        """
        self.masks = yolo_masks
        anchors = torch.Tensor(yolo_anchors)
        self.anchors = list()
        for i in yolo_masks:
            self.anchors.append(anchors[i[0]:i[2]+1])

        # print(self.anchors)

        self.na = len(self.anchors)  # number of anchors (3)

        self.nc = num_classes  # number of classes (80)
        self.no = self.nc + 5  # number of outputs (85)
        self.stride = stride
        self.anchor_vec = list()
        self.anchor_wh = list()

        for i in range(len(self.anchors)):
            self.anchor_vec.append(self.anchors[i] / self.stride[i])
            self.anchor_wh .append( self.anchor_vec[i].view(1, self.na, 1, 1, 2))
        self.grid = list()

    def process(self, outputs):

        outputs_reshaped = list()
        self.grid = list()
        for output in outputs:

            bs, _, ny, nx = output.shape
            self.create_grids((nx, ny))
            # print('shape bf:',output.shape)
            # print(output)
            # print('-'*80)
            tmp = torch.from_numpy(output)
            tmp = tmp.view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()  # prediction
            # print(tmp.shape)
            # print((tmp))
            # print("="*40,'\n\n')
            outputs_reshaped.append(tmp)
            # print('shape af:',self._reshape_output(output).shape)

        # print('len of outputs_reshaped: ', len(outputs_reshaped)) # 3 for yolov3
        batch_size = outputs_reshaped[0].shape[0]
        
        i = 0
        output_list = list()
        for output in (outputs_reshaped):

            output[..., :2]  = torch.sigmoid(output[..., :2]) + self.grid[i]  # xy
            output[..., 2:4] = torch.exp(output[..., 2:4]) * self.anchor_wh[i]  # wh yolo method
            output[..., :4] *= self.stride[i]
            i = i + 1

            torch.sigmoid_(output[..., 4:])
            output_list.append( output.view(batch_size, -1, self.no))  # view [1, 3, 13, 13, 85] as [1, 507, 85]
            # print(output.view(batch_size, -1, self.no).shape)
            # print((output.view(batch_size, -1, self.no)))
            # print("="*40)

        return output_list

    def create_grids(self, ng=(13, 13)):
        self.nx, self.ny = ng  # x and y grid size
        self.ng = torch.tensor(ng, dtype=torch.float)

        yv, xv = torch.meshgrid([torch.arange(self.ny), torch.arange(self.nx)])
        self.grid.append(torch.stack((xv, yv), 2).view((1, 1, self.ny, self.nx, 2)).float())




