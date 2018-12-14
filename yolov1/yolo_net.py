import tensorflow as tf
import numpy as np
import re
from .net import Net


class YoloNet(Net):
    def __init__(self,common_params,net_params,test=False):
        super().__init__(common_params,net_params)
        self.image_size=int(common_params['image_size'])
        self.num_classes=int(common_params['num_classes'])
        self.cell_size=int(net_params['cell_size'])
        self.boxes_per_cell=int(net_params['boxes_per_cell'])
        self.batch_size=int(common_params['batch_size'])
        self.weighted_decay=float(net_params['weight_decay'])

        if not test:
            self.object_scale=float(net_params['object_scale'])
            self.noobject_scale=float(net_params['noobject_scale'])
            self.class_scale=float(net_params['class_scale'])
            self.coord_scale=float(net_params['coord_scale'])




    def inference(self,images):
        conv_num=1
        temp_conv=self.conv2d('conv'+str(conv_num),images,[7,7,3,64],stride=2)
        conv_num+=1

        temp_pool=self.max_pool(temp_conv,[2,2],2)
        temp_conv=self.conv2d('conv'+str(conv_num),temp_pool,[3,3,64,192],stride=1)
        conv_num+=1

        temp_pool=self.max_pool(temp_conv,[2,2],2)
        temp_conv=self.conv2d('conv'+str(conv_num),temp_pool,[1,1,192,128],stride=1)
        conv_num+=1

        temp_conv = self.conv2d('conv' + str(conv_num), temp_conv, [3, 3, 128, 256], stride=1)
        conv_num+=1

        temp_conv = self.conv2d('conv' + str(conv_num), temp_conv, [1, 1, 256, 256], stride=1)
        conv_num += 1

        temp_conv = self.conv2d('conv' + str(conv_num), temp_conv, [3, 3, 256, 512], stride=1)
        conv_num += 1

        temp_conv=self.max_pool(temp_conv,[2,2],2)

        for i in range(4):
            temp_conv=self.conv2d('conv'+str(conv_num),temp_conv,[1,1,512,256],stride=1)
            conv_num+=1
            temp_conv = self.conv2d('conv' + str(conv_num), temp_conv, [3,3,256,512], stride=1)
            conv_num += 1

        temp_conv = self.conv2d('conv' + str(conv_num), temp_conv, [1, 1, 512, 512], stride=1)
        conv_num += 1
        temp_conv = self.conv2d('conv' + str(conv_num), temp_conv, [3, 3, 512, 1024], stride=1)
        conv_num += 1

        temp_conv=self.max_pool(temp_conv,[2,2],2)

        for i in range(2):
            temp_conv=self.conv2d('conv'+str(conv_num),temp_conv,[1,1,1024,512],stride=1)
            conv_num+=1
            temp_conv=self.conv2d('conv'+str(conv_num),temp_conv,[3,3,512,1024],stride=1)
            conv_num+=1

        temp_conv=self.conv2d('conv'+str(conv_num),temp_conv,[3,3,1024,1024],stride=1)
        conv_num+=1
        temp_conv=self.conv2d('conv'+str(conv_num),temp_conv,[3,3,1024,1024],stride=2)
        conv_num+=1


        temp_conv=self.conv2d('conv'+str(conv_num),temp_conv,[3,3,1024,1024],stride=1)
        conv_num+=1
        temp_conv = self.conv2d('conv' + str(conv_num), temp_conv, [3, 3, 1024, 1024], stride=1)
        conv_num += 1



        local1=self.local('local1',temp_conv,49*1024,4096)
        local1=tf.nn.dropout(local1,keep_prob=0.5)

        local2=self.local('local2',local1,4096,self.cell_size*self.cell_size*(self.num_classes+5*self.boxes_per_cell),leaky=False)

        local2=tf.reshape(local2,[tf.shape(local2)[0],self.cell_size,self.cell_size,self.num_classes+5*self.boxes_per_cell])
        predicts=local2
        return predicts

    def iou(self,boxes1,boxes2):
        """

        :param boxes1:[cell,cell,boxes_per_cell,4]==>(x_center,y_center,w,h),4D tensor
        :param boxes2: 1-D tensor with (x_center,y_center,w,h)
        :return: 3-D tensor[CELL_SIZE,CELL_SIZE,BOXES_PER_CELL]
        """
        #transfer to topleft(x,y) & bottomright(x,y)
        tf.stack(
          [boxes1[:,:,:,0]-boxes1[:,:,:,2]/2,boxes1[:,:,:,1]-boxes1[:,:,:,3]/2,
           boxes1[:,:,:,0]+boxes1[:,:,:,2]/2,boxes1[:,:,:,1]-boxes1[:,:,:,3]/2]
        )


    def loss(self,predicts,labels,objects_num):
        pass


