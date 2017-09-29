'''
Deep Learning Programming Assignment 2
--------------------------------------
Name: Ravi Bansal
Roll No.: 13CS30026

======================================
Complete the functions in this file.
Note: Do not change the function signatures of the train
and test functions
'''
import numpy as np
import tensorflow as tf
import numpy as np
import os
import shutil

import urllib2
import zipfile

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')


def train(trainX, trainY):
    '''
    Complete this function.
    '''
    input_dim = trainX[0].flatten().shape[0]
    output_dim = 10
    
    num_inputs = trainX.shape[0]
    trainX=trainX.reshape(num_inputs,input_dim)

    sess = tf.InteractiveSession()
    x = tf.placeholder(tf.float32,[None,input_dim])
    y_ = tf.placeholder(tf.float32,[None,output_dim])

    W_conv1 = weight_variable([5, 5, 1, 32])
    b_conv1 = bias_variable([32])

    x_image = tf.reshape(x, [-1,28,28,1])

    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])

    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

    W_fc1 = weight_variable([7 * 7 * 64, 1024])
    b_fc1 = bias_variable([1024])

    h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
    
    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    W_fc2 = weight_variable([1024, 10])
    b_fc2 = bias_variable([10]) 

    y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    
    sess.run(tf.global_variables_initializer())
    
    batch_size = 50

    for k in range(17):
        perm = np.random.permutation(trainX.shape[0])
        trainX, trainY = trainX[perm], trainY[perm]
        
        for i in xrange(0,num_inputs,batch_size):
            trainX_batch = trainX[i:i+batch_size]
            trainY_batch = trainY[i:i+batch_size] # 100
            output_batch = np.zeros((batch_size,output_dim)) 
            for j in xrange(0,batch_size): 
                output_batch[j][trainY_batch[j]]=1
            if ((k*num_inputs + i)/batch_size)%100 == 0:
                train_accuracy = accuracy.eval(
                    feed_dict={x:trainX_batch, y_: output_batch, keep_prob: 1.0})
                print("step %d, training accuracy %g"%((
                    k*num_inputs + i)/batch_size, train_accuracy))
            train_step.run(feed_dict={x: trainX_batch, 
                y_: output_batch, keep_prob: 0.5})

    saver = tf.train.Saver()
    tf.add_to_collection('x', x)
    tf.add_to_collection('y_conv', y_conv)
    tf.add_to_collection('keep_prob', keep_prob)
    if not os.path.exists('saved_model_cnn'):
        os.makedirs('saved_model_cnn')
    else:
        shutil.rmtree('saved_model_cnn')
        os.makedirs('saved_model_cnn')
    saver.save(sess, 'saved_model_cnn/trained-model')
    sess.close()
    tf.reset_default_graph()

def test(testX):
    '''
    Complete this function.
    This function must read the weight files and
    return the predicted labels.
    The returned object must be a 1-dimensional numpy array of
    length equal to the number of examples. The i-th element
    of the array should contain the label of the i-th test
    example.
    '''
    num_inputs=len(testX)
    input_dim = testX[0].flatten().shape[0]
    testX=testX.reshape(num_inputs,input_dim)

    if not os.path.exists('saved_model_cnn'):
        print "Downloading saved weights for CNN to saved_model_cnn ......"
        proxy = urllib2.ProxyHandler({'https': '10.3.100.207:8080'})
        opener = urllib2.build_opener(proxy)
        urllib2.install_opener(opener)

        with open('saved_model_cnn.zip','wb') as f:
            f.write(urllib2.urlopen(
                "https://github.com/ravibansal/saved_files/raw/master/saved_model_cnn.zip").read())
            f.close()

        zip_ref = zipfile.ZipFile('./saved_model_cnn.zip', 'r')
        zip_ref.extractall('./')
        zip_ref.close()

    sess = tf.Session()
    new_saver = tf.train.import_meta_graph('saved_model_cnn/trained-model.meta')
    new_saver.restore(sess, tf.train.latest_checkpoint('./saved_model_cnn/'))
    x = tf.get_collection('x')[0]
    y_conv = tf.get_collection('y_conv')[0]
    keep_prob = tf.get_collection('keep_prob')[0]
    
    y = sess.run(y_conv, feed_dict={x:testX, keep_prob: 1.0})

    labels = np.argmax(y,axis=1)

    sess.close()
    tf.reset_default_graph()
    
    return labels
