import tensorflow as tf
import numpy as np

class vgg16(object):
    def __init__(self, num_classes = 200, train= 1,  weight_path=None, param_list1=['fc8'], param_list2 = ['conv1','conv2','conv3','conv4','conv5','fc6','fc7']):
        self.filter_size = 3
        self.num_classes = num_classes
        self.train  = train
        self.param = []
        self.param_finetune  =[]
        self.param_train_complete = []
        self.height = 224
        self.width = 224
        self.param_list1 = param_list1
        self.param_list2 = param_list2

        #define placeholders
        self.x = tf.placeholder(tf.float32, shape=[None, self.height, self.width, 3])
        self.y = tf.placeholder(tf.int32, shape=[None, self.num_classes])

        #build the model
        self.build()


    def build(self):
        temp = self.conv(self.x)
        self.logits = self.fc(temp)

        #cross-entopy loss -function with softmax 
        self.loss_ = tf.nn.softmax_cross_entropy_with_logits(labels=self.y , logits = self.logits)
        self.loss = tf.reduce_mean(self.loss_)


    def optimize(self, ):

        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        
        self.train_op1 = tf.train.AdamOptimizer(learning_rate=0.0001)
        self.train_op2 = tf.train.AdamOptimizer(learning_rate=0.001)

        print('tf non-trainable variable list')
        for var in self.param:
            print(var, var.name)
        print('')   
        print('tf fine-tuning parameter list')
        for var in self.param_finetune:
            print(var, var.name)
        print('')          
        print('tf fully-trainable parameter list')
        for var in self.param_train_complete:
            print(var, var.name)
        print('')
        
	#compute gradients
        self.grads = tf.gradients(self.loss, self.param_finetune + self.param_train_complete)
        self.grads1 = self.grads[:len(self.param_finetune)]
        self.grads2 = self.grads[len(self.param_finetune):]

        #update weights with help of gradients
        self.optimizer1 = self.train_op1.apply_gradients(zip(self.grads1, self.param_finetune))
        self.optimizer2 = self.train_op2.apply_gradients(zip(self.grads2, self.param_train_complete), global_step=self.global_step)

        #group this update into a single variable
        self.optimizer = tf.group(self.optimizer1, self.optimizer2)

    def conv(self, x):
        conv1_1 = self.conv_layer(x, 3, 64 ,'conv1_1')
        conv1_2 = self.conv_layer(conv1_1, 64, 64 ,'conv1_2')
        pool1 = self.max_pool(conv1_2, 'pool1')

        conv2_1 = self.conv_layer(pool1, 64, 128 ,'conv2_1')
        conv2_2 = self.conv_layer(conv2_1, 128, 128,'conv2_2')
        pool2 = self.max_pool(conv2_2, 'pool2')

        conv3_1 = self.conv_layer(pool2, 128, 256,'conv3_1')
        conv3_2 = self.conv_layer(conv3_1, 256, 256,'conv3_2')
        conv3_3 = self.conv_layer(conv3_2, 256, 256,'conv3_3')
        pool3 = self.max_pool(conv3_3, 'pool3')

        conv4_1 = self.conv_layer(pool3, 256, 512,'conv4_1')
        conv4_2 = self.conv_layer(conv4_1, 512, 512,'conv4_2')
        conv4_3 = self.conv_layer(conv4_2, 512, 512,'conv4_3')
        pool4 = self.max_pool(conv4_3, 'pool4')

        conv5_1 = self.conv_layer(pool4, 512, 512,'conv5_1')
        conv5_2 = self.conv_layer(conv5_1, 512, 512,'conv5_2')
        conv5_3 = self.conv_layer(conv5_2, 512, 512,'conv5_3')
        pool5 = self.max_pool(conv5_3, 'pool5')

        return pool5

    def fc(self, x):
        fc6 = self.fc_layer(x, 7*7*512, 4096,'fc6')
        fc7 = self.fc_layer(fc6, 4096, 4096,'fc7')
        fc8 = self.fc_layer(fc7, 4096, self.num_classes,'fc8', pred_layer= True)
        self.pred = tf.argmax(tf.nn.softmax(fc8), axis = 1)
        return fc8

    def conv_layer(self, x, in_channels, out_channels, name):
        filt = tf.Variable(tf.truncated_normal(shape=[self.filter_size, self.filter_size, in_channels, out_channels ], stddev = 0.01), name = name + '_W')
        bias = tf.Variable(tf.constant([0.1], shape=[out_channels]), name = name + '_b')      
        if self.train == 2:
            self.param_train_complete += [filt, bias]
        elif self.train == 1:
            if name.split('_')[0] in self.param_list1:
                self.param_train_complete += [filt, bias]
            elif name.split('_')[0] in self.param_list2:
                self.param_finetune += [filt, bias]
            else:
                self.param += [filt, bias]
        else:
            self.param += [filt, bias]
        
        return self.relu(tf.nn.bias_add(tf.nn.conv2d( x, filt, strides = [ 1,1,1,1], padding ='SAME', data_format='NHWC'),bias))

    def fc_layer(self, x, in_channels, out_channels, name, pred_layer= False):
        x = tf.reshape(x, [-1, in_channels])
        weights = tf.get_variable(name = name + '_W', shape=[ in_channels, out_channels ], initializer=tf.contrib.layers.xavier_initializer() )
        bias = tf.Variable(tf.constant([0.1], shape=[out_channels]), name = name + '_b')
        
        if pred_layer:
            self.param_train_complete += [weights, bias]
            return tf.nn.bias_add(tf.matmul(x, weights), bias) 
                
        if self.train == 2:
            self.param_train_complete += [weights, bias]
            return tf.nn.dropout(self.relu(tf.nn.bias_add(tf.matmul( x, weights), bias)), self.dropout)
        elif self.train == 1:
            if name in self.param_list1:
                self.param_train_complete += [weights, bias]
            elif name in self.param_list2:
                self.param_finetune += [weights, bias]
            else:
                self.param += [weights, bias]
            return self.relu(tf.nn.bias_add(tf.matmul( x, weights), bias))
        else:
            self.param += [weights, bias]
            return self.relu(tf.nn.bias_add(tf.matmul(x, weights), bias))
        
    def max_pool(self, x, name):
        return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME', data_format='NHWC', name = name)

    def relu(self, x):
        return tf.nn.relu(x)

    def load_weight(self,sess, weight_path ):
        pre_trained_weights = np.load(weight_path)
        keys = sorted(pre_trained_weights.keys())
        num_non_trainable_param = len(self.param)
        num_fine_tuning_param = len(self.param_finetune)
        if self.train != 2:
            for i, k in enumerate(keys):
                if  i < 30   :
                    print(i , k )
                    if i < num_non_trainable_param:
                        sess.run(self.param[i].assign(pre_trained_weights[k]))
                    elif i < num_fine_tuning_param + num_non_trainable_param:
                        sess.run(self.param_finetune[i - num_non_trainable_param].assign(pre_trained_weights[k]))
                    else:
                        sess.run(self.param_train_complete[i - num_fine_tuning_param - num_non_trainable_param].assign(pre_trained_weights[k]))
