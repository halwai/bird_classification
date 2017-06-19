import tensorflow as tf
import numpy as np

class alex(object):
    def __init__(self, num_classes = 200, dropout=0.5, train= 1,  weight_path=None, param_list1=['fc8'], param_list2 = ['fc6','fc7']):
        self.dropout = dropout
        self.num_classes = num_classes
        self.train  = train
        self.param = []
        self.param_finetune  =[]
        self.param_train_complete = []
        self.height = 227
        self.width = 227
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
        
        self.train_op1 = tf.train.AdamOptimizer(learning_rate=0.001)
        self.train_op2 = tf.train.AdamOptimizer(learning_rate=0.01)

        #compute gradients
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
        self.grads = tf.gradients(self.loss, self.param_finetune + self.param_train_complete)
        self.grads1 = self.grads[:len(self.param_finetune)]
        self.grads2 = self.grads[len(self.param_finetune):]

        #update weights with help of gradients
        self.optimizer1 = self.train_op1.apply_gradients(zip(self.grads1, self.param_finetune))
        self.optimizer2 = self.train_op2.apply_gradients(zip(self.grads2, self.param_train_complete), global_step=self.global_step)

        #group this update into a single variable
        self.optimizer = tf.group(self.optimizer1, self.optimizer2)

    def conv(self, x):
        conv1 = self.conv_layer(x, 3, 96 ,11 ,4, 'conv1', groups = 1, padding = 'SAME')
        lrn1 = tf.nn.lrn(conv1, 2, 2, 1e-4, 0.75,'norm1')
        pool1 = self.max_pool(lrn1, 'pool1')
        
        conv2 = self.conv_layer(pool1, 96, 256, 5 , 1, 'conv2')
        lrn2 = tf.nn.lrn(conv2, 2, 2, 1e-4, 0.75,'norm2')
        pool2 = self.max_pool(lrn2, 'pool2')

        conv3 = self.conv_layer(pool2, 256, 384, 3, 1, 'conv3', groups = 1)

        conv4 = self.conv_layer(conv3, 384, 384, 3, 1, 'conv4')

        conv5 = self.conv_layer(conv4, 384, 256, 3, 1, 'conv5')
        pool5 = self.max_pool(conv5, 'pool5')
        
        #print(conv1, lrn1, pool1 , conv2, lrn2 , pool2 , conv3, conv4, conv5, pool5)

        return pool5

    def fc(self, x):
        fc6 = self.fc_layer(x, 6*6*256, 4096,'fc6')
        fc7 = self.fc_layer(fc6, 4096, 4096,'fc7')
        fc8 = self.fc_layer(fc7, 4096, self.num_classes,'fc8', pred_layer= True)
        self.pred = tf.argmax(tf.nn.softmax(fc8), axis = 1)
        return fc8

    def conv_layer(self, x, in_channels, out_channels, filter_size, stride, name, groups = 2, padding='SAME'):
        filt = tf.Variable(tf.truncated_normal(shape=[filter_size, filter_size, int(in_channels/groups), out_channels ], stddev = 0.01), name = name + '_W')
        bias = tf.Variable(tf.constant([0.1], shape=[out_channels]), name = name + '_b')      
        if self.train == 2:
            self.param_train_complete += [filt, bias]
        elif self.train == 1:
            if name in self.param_list1:
                self.param_train_complete += [filt, bias]
            elif name in self.param_list2:
                self.param_finetune += [filt, bias]
            else:
                self.param += [filt, bias]
        else:
            self.param += [filt, bias]
        if groups == 1:
            return self.relu(tf.nn.bias_add(tf.nn.conv2d( x, filt, strides = [ 1,stride,stride,1], padding = padding, data_format='NHWC'),bias))
        else:
            # Split input and weights and convolve them separately
            input_groups = tf.split(axis = 3, num_or_size_splits=groups, value=x)
            filt_groups = tf.split(axis = 3, num_or_size_splits=groups, value=filt)
            output_groups = [ tf.nn.conv2d( i, k, strides = [ 1,stride,stride,1], padding = padding, data_format='NHWC') for i,k in zip(input_groups, filt_groups)]

            conv = tf.concat(axis = 3, values = output_groups)

            return self.relu(tf.nn.bias_add(conv, bias))


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
        return tf.nn.max_pool(x, ksize=[1,3,3,1], strides=[1,2,2,1], padding='VALID', data_format='NHWC', name = name)

    def relu(self, x):
        return tf.nn.relu(x)

    def load_weight(self,sess, weight_path ):
        pre_trained_weights = np.load(open(weight_path, "rb"), encoding="latin1").item()
        keys = sorted(pre_trained_weights.keys())
        num_non_trainable_param = len(self.param)
        num_fine_tuning_param = len(self.param_finetune)
        if self.train != 2:
            i = 0
            for k in keys:
                if  i < 14:
                    print(i , k )
                    if i < num_non_trainable_param:
                        sess.run(self.param[i].assign(pre_trained_weights[k][0]))
                        i += 1
                        sess.run(self.param[i].assign(pre_trained_weights[k][1]))
                    elif i < num_fine_tuning_param + num_non_trainable_param:
                        sess.run(self.param_finetune[i - num_non_trainable_param].assign(pre_trained_weights[k][0]))
                        i += 1
                        sess.run(self.param_finetune[i - num_non_trainable_param].assign(pre_trained_weights[k][1]))
                    else:
                        sess.run(self.param_train_complete[i - num_fine_tuning_param - num_non_trainable_param].assign(pre_trained_weights[k][0]))
                        i += 1
                        sess.run(self.param_train_complete[i - num_fine_tuning_param - num_non_trainable_param].assign(pre_trained_weights[k][1]))

                    i += 1
