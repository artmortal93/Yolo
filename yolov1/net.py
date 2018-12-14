import tensorflow as tf
from easydict import EasyDict as edict
import numpy


class Net(object):
    def __init__(self,commmon_params,net_params):
        self.pretrained_collection=[]
        self.trainable_collection=[]

    def _variable_on_cpu(self,name,shape,initializer,pretrain=True,train=True):
        with tf.device('/cpu:0'):
            var=tf.get_variable(name,shape,initializer=initializer,dtype=tf.float32)
            if pretrain==True:
                self.pretrained_collection.append(var)
            if train==True:
                self.trainable_collection.append(var)
        return var

    def _variable_with_weight_decay(self,name,shape,stddev,wd,pretrain=True,train=True):
        """

        :param name:
        :param shape:
        :param stddev:
        :param wd:short for weight decay
        :param pretrain:
        :param train:
        :return:
        """
        var=self._variable_on_cpu(name,shape,initializer=tf.truncated_normal_initializer(stddev=stddev,dtype=tf.float32),pretrain=pretrain,train=train)
        if wd is not None:
            weight_decay=tf.multiply(tf.nn.l2_loss(var),wd,name='weight_loss')
            tf.add_to_collection('losses',weight_decay)
        return var

    def conv2d(self,scope,input,kernel_size,stride=1,pretrain=True,train=True):
        with tf.variable_scope(scope) as scope:
            kernel=self._variable_with_weight_decay('weights',shape=kernel_size,stddev=5e-2,wd=self.weight_decay,
                                                    pretrain=pretrain,train=train)
            conv=tf.nn.conv2d(input,kernel,[1,stride,stride,1],padding='SAME')
            biases=self._variable_on_cpu('biases',kernel_size[3:],tf.constant_initializer(0.0),pretrain,train)
            bias=tf.nn.bias_add(conv,biases)
            conv1=self.leaky_relu(bias)
            return conv1


    def max_pool(self,input,kernel_size,stride):
        return tf.nn.max_pool(input,ksize=[1,kernel_size[0],kernel_size[1],1],strides=[1,stride,stride,1],padding='SAME')


    def local(self,scope,input,in_dimension,out_dimension,leaky=True,pretrain=True,train=True):
        with tf.variable_scope(scope) as scope:
            reshape=tf.reshape(input,[tf.shape(input)[0],-1])
            weights=self._variable_with_weight_decay('weights',shape=[in_dimension,out_dimension],
                                                     stddev=0.04,wd=self.weight_decay,pretrain=pretrain,train=train)
            biases=self._variable_on_cpu('biases',[out_dimension],tf.constant_initializer(0.0),pretrain,train)
            local=tf.matmul(reshape,weights)+biases

            if leaky:
                local=self.leaky_relu(local)
            else:
                local=tf.identity(local,name=scope.name)
            return local


    def leaky_relu(self,x,alpha=0.1,dtype=tf.float32):
        x=tf.cast(x,dtype=dtype)
        bool_mask=(x>0)
        mask=tf.cast(bool_mask,dtype=dtype)
        return 1.0*mask*x+alpha*(1-mask)*x



    def inference(self,images):
        """

        :param images:
        :return: 4D TENSOR with[batch,cell,cell.num_classes+5*boxes_per_cell]
        """
        raise NotImplementedError

    def loss(self, predicts, labels, objects_num):

        """Add Loss to all the trainable variables
        Args:
          predicts: 4-D tensor [batch_size, cell_size, cell_size, 5 * boxes_per_cell]
          ===> (num_classes, boxes_per_cell, 4 * boxes_per_cell)
          labels  : 3-D tensor of [batch_size, max_objects, 5]
          objects_num: 1-D tensor [batch_size]
        """
        raise NotImplementedError

