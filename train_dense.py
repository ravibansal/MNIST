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

def train(trainX, trainY):
    '''
    Complete this function.
    '''
    batch_size=100
    input_dim = trainX[0].flatten().shape[0]
    hidden_dim = 300
    output_dim = 10

    num_inputs = trainX.shape[0]
    trainX=trainX.reshape(num_inputs,input_dim) #60000 x 784
    # print num_inputs #60000
    

    sess = tf.InteractiveSession()
    X = tf.placeholder(tf.float32,[None,input_dim])
    Y_ = tf.placeholder(tf.float32,[None,output_dim])
    W1 = tf.Variable(tf.random_uniform([input_dim,hidden_dim],
        -(1.0/input_dim)**(1/2.0),(1.0/input_dim)**(1/2.0),tf.float32))
    b1 = tf.Variable(tf.zeros([1,hidden_dim]))
    W2 = tf.Variable(tf.random_uniform([hidden_dim,output_dim],
        -(1.0/hidden_dim)**(1/2.0),(1.0/hidden_dim)**(1.0/2),tf.float32))
    b2 = tf.Variable(tf.zeros([1,output_dim]))
    Z1 = tf.matmul(X,W1) + b1
    Y = tf.matmul(tf.nn.relu(Z1),W2) + b2

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y_, 
        logits=Y))/batch_size
    train_step = tf.train.GradientDescentOptimizer(0.4).minimize(loss)
    sess.run(tf.global_variables_initializer())

    for k in xrange(20):
        perm = np.random.permutation(trainX.shape[0])
        trainX, trainY = trainX[perm], trainY[perm]

        for i in xrange(0,num_inputs,batch_size):
            trainX_batch = trainX[i:i+batch_size]
            trainY_batch = trainY[i:i+batch_size] # 100
            output_batch = np.zeros((batch_size,output_dim)) 
            for j in xrange(0,batch_size): 
                output_batch[j][trainY_batch[j]]=1
            train_step.run(feed_dict={X:trainX_batch, Y_:output_batch})
    saver = tf.train.Saver()
    tf.add_to_collection('X', X)
    tf.add_to_collection('Y', Y)
    if not os.path.exists('saved_model'):
        os.makedirs('saved_model')
    else:
        shutil.rmtree('saved_model')
        os.makedirs('saved_model')
    saver.save(sess, 'saved_model/trained-model')
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

    if not os.path.exists('saved_model'):
        print "Downloading saved weights for DNN to saved_model ......"
        proxy = urllib2.ProxyHandler({'https': '10.3.100.207:8080'})
        opener = urllib2.build_opener(proxy)
        urllib2.install_opener(opener)

        with open('saved_model.zip','wb') as f:
            f.write(urllib2.urlopen(
                "https://github.com/ravibansal/saved_files/raw/master/saved_model.zip").read())
            f.close()

        zip_ref = zipfile.ZipFile('./saved_model.zip', 'r')
        zip_ref.extractall('./')
        zip_ref.close()

    sess = tf.Session()
    new_saver = tf.train.import_meta_graph('saved_model/trained-model.meta')
    new_saver.restore(sess, tf.train.latest_checkpoint('./saved_model/'))
    X = tf.get_collection('X')[0]
    Y = tf.get_collection('Y')[0]
    y = sess.run(Y, feed_dict={X:testX})
    labels = np.argmax(y,axis=1)
    sess.close()
    tf.reset_default_graph()
    
    return labels
