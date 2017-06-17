from __future__ import division
from __future__ import print_function

#import important modules
import tensorflow as tf
import numpy as np
import argparse
import time

#import models
from vgg16 import vgg16

#import data generator/provider
from bird_dataset_generator import BirdClassificationGenerator

#import utility functions
from utils import get_batch, save_csv


class deep_neural_net(object):
    def __init__(self, dataset_dir, batch_size=100 , val_ratio= 0.2, num_epochs = 1001,  gpu_memory_fraction = 0.4, model_name='my_vgg', model_weight_path = None):
        self.batch_size = batch_size 
        self.val_ratio  = val_ratio
        self.num_epochs  = num_epochs

        self.obj = BirdClassificationGenerator(dataset_dir, self.val_ratio, self.batch_size)

        #load the model
        if model_name == 'vgg':
            self.model = vgg16(weight_path=model_weight_path, train = 1)


        self.run_training_testing(model_weight_path, gpu_memory_fraction)



    def evaluate(self, sess, set_type):
        if set_type == 'val':
            num_images = len(self.obj.val_list)
            generator = self.obj.val_generator()
        else:
            num_images = len(self.obj.train_list)
            generator = self.obj.train_generator()
        
        true_positives = 0
        val_loss = 0
        num_batches = num_images//self.batch_size if num_images%self.batch_size == 0 else num_images//self.batch_size + 1 
        for i in range(num_batches):
            x_batch, y_batch = get_batch(generator, set_type, height=self.model.height, width=self.model.width)

            predicted = sess.run([self.model.pred], feed_dict={self.model.x:x_batch, self.model.y:y_batch})
            
            true_positives = true_positives + np.sum(predicted[0] == np.argmax(y_batch,1))

        print('set_type:',set_type, 'accuracy = ', true_positives*100.0/num_images)
        

    #predict the labels for test dataset
    def predict(self, sess, set_type):
        if set_type == 'val':
            num_images = len(self.obj.val_list)
            generator = self.obj.val_generator()
        elif  set_type == 'test':
            num_images = len(self.obj.test_list)
            generator = self.obj.test_generator()
        else:
            num_images = len(self.obj.train_list)
            generator = self.obj.train_generator()
        
        true_positives = 0
        num_batches = num_images//self.batch_size if num_images%self.batch_size == 0 else num_images//self.batch_size + 1 
        model_predictions = []
        for i in range(num_batches):
            x_batch, _ = get_batch(generator, set_type , height=self.model.height, width=self.model.width)
            predicted = sess.run([ self.model.pred], feed_dict={self.model.x:x_batch})
            model_predictions.extend(predicted[0])
        return model_predictions
    

    def run_training_testing(self, model_weight_path, gpu_memory_fraction):

        # train the network
        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = gpu_memory_fraction

        train_generator_obj = self.obj.train_generator()

        with tf.Session(config=config) as sess:
            summary_writer = tf.summary.FileWriter('./checkpoints/', sess.graph)
            saver = tf.train.Saver(max_to_keep=2)
            self.model.optimize()           

            sess.run(tf.global_variables_initializer())

            self.model.load_weight(sess, model_weight_path)
            
            loss = 0
            true_positives = 0
            for epochs in range(1, self.num_epochs+1):
                start_time = time.time()
                for step in range(len(self.obj.train_list)//self.batch_size + 1):
                    x_batch, y_batch = get_batch(train_generator_obj, 'train', height=self.model.height, width=self.model.width)
                    _, loss_curr, predicted = sess.run([self.model.optimizer, self.model.loss, self.model.pred] , feed_dict={self.model.x:x_batch, self.model.y:y_batch})
                    loss = 0.9*loss + 0.1*loss_curr
                    true_positives = true_positives + np.sum(predicted == np.argmax(y_batch,1))

                end_time = time.time()
                print('time_taken', end_time -start_time)    
                print('epochs:',epochs, ' train-loss:', loss, 'train-acc:', true_positives*100.0/len(self.obj.train_list))                    
                true_positives = 0

                saver.save(sess, './checkpoints/', global_step=step)
                self.evaluate(sess, 'val')
                print('')


        # predict values for test dataset
        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = gpu_memory_fraction
        
        with tf.Session(config=config) as sess:
            saver.restore(sess, tf.train.latest_checkpoint('./checkpoints/'))
            model_pred = self.predict(sess, 'test')    


        #save the results in the required csv format
        save_csv(model_pred, self.obj)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument( "--batch_size", help="batch_size",         default = 100, type=int)
    parser.add_argument( "--val_ratio",  help="validation-ratio",   default = 0.2, type=float)
    parser.add_argument( "--num_epochs",   help="num of iterations",  default = 1000, type=int)
    parser.add_argument( "--server",     help="code on server?",    default = 1,    type=int)
    parser.add_argument( "--gpu_mem_frac",   help="% of gpu-mem",   default = 0.4, type=float)
    parser.add_argument( "--model",      help="model-name = vgg|alex|resnet", required=True , type=str)

    config = parser.parse_args()
    if config.server == 0:
        dataset_dir = '/home/halwai/coursework/deep_learning_course/week2/assignment1/CUB_200_2011/'
        vgg_weight_path = '/home/halwai/Downloads/vgg16_weights.npz'
    else:
        dataset_dir = '/data4/abhijeet/Datasets/CUB_200_2011/'
        vgg_weight_path = '/data4/abhijeet/vgg16_weights.npz'    

    net = deep_neural_net(dataset_dir, config.batch_size, config.val_ratio, config.num_epochs, config.gpu_mem_frac, config.model, vgg_weight_path)
