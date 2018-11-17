#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
import sys
import os.path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('data_dir', 'data', 'Input Directory.')
flags.DEFINE_string('output_dir', 'data', 'Output Directory.')

def main(_):
    import pandas as pd
    print(FLAGS.data_dir)
    train_input = pd.read_csv(os.path.join(FLAGS.data_dir, 'X_train.csv'))
    train_output = pd.read_csv(os.path.join(FLAGS.data_dir, 'y_train.csv'))
    test_input = pd.read_csv(os.path.join(FLAGS.data_dir, 'X_test.csv'))


    # In[3]:


    num_of_train_samples = len(train_input)
    num_of_features = len(train_input.loc[0]) - 1 # first one is for id
    print( num_of_features)
    print( num_of_train_samples)


    # In[4]:


    print(train_input.describe())
    print(train_output.describe())


    # In[5]:


    train_input


    # In[6]:


    print(train_input['id'].dtype)
    train_input = train_input.sort_values(by=['id'])
    train_input = train_input.drop(columns=['id'])
    train_output = train_output.sort_values(by=['id'])
    train_output = train_output.drop(columns=['id'])
    test_input = test_input.sort_values(by=['id'])
    test_input = test_input.drop(columns=['id'])


    # In[7]:


    # Seeing whats the total number of NaNs per feature
    ((train_input.isna().sum() / num_of_features * 100) > 20).sum()


    # In[17]:


    import matplotlib.pyplot as plt
    train_input.loc[0][:8851].plot()
    # plt.show()


    # In[20]:


    train_input.loc[256][: 8602].plot()
    # plt.show()


    # In[22]:


    average_per_feature = train_input.mean()
    train_input_fill1 = train_input.fillna(average_per_feature)
    train_input_fill1.loc[256][:].plot()
    # plt.show()


    # In[26]:


    train_input_fill1 = train_input.interpolate(method='linear', axis=1)
    train_input_fill1.loc[8][:].plot()
    # plt.show()
    train_input.loc[8][:].plot()
    # plt.show()


    # In[27]:


    #THE second filling NANs seems like a good one, which is linear interpolation
    train_input = train_input.interpolate(method='linear', axis=1)
    test_input = test_input.interpolate(method='linear', axis=1)
    train_output = train_output.squeeze()


    # In[28]:


    train_output


    # In[29]:


    original_train_output_shape = train_output.shape
    original_train_input_shape = train_input.shape
    original_test_input_shape = test_input.shape
    print(train_output.shape)
    print(train_input.shape)
    print(test_input.shape)


    # In[36]:


    #NORMALIZING PARAMS
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(train_input, train_output, test_size=0.2)
    print(X_train.shape)
    print(X_test.shape)


    # In[51]:


    #SCALING
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)


    # In[52]:


    #Training Parameters
    learning_rate = 0.001
    num_steps = 2000
    batch_size = 256

    # Network Parameters
    num_input = 18154 # MNIST data input (img shape: 28*28)
    num_classes = 4 # MNIST total classes (0-9 digits)
    dropout = 0.25 # Dropout, probability to drop a unit


    # In[53]:


    import tensorflow as tf


    # In[54]:


    # Create the neural network
    def conv_net(x_dict, n_classes, dropout, reuse, is_training):
        
        # Define a scope for reusing the variables
        with tf.variable_scope('ConvNet', reuse=reuse):
            # TF Estimator input is a dict, in case of multiple inputs
            x = x_dict['images']
            
            # Tensor input become 4-D: [Batch Size, Height, Width, Channel]
            x = tf.reshape(x, shape=[-1, 18154, 1])
            conv1 = tf.layers.conv1d(x, 128, 5, activation=tf.nn.relu)
            # Max Pooling (down-sampling) with strides of 2 and kernel size of 2
            conv1 = tf.layers.max_pooling1d(conv1, 2, 2)
            conv1 = tf.layers.dropout(conv1, rate=dropout, training=is_training)

            # Convolution Layer with 64 filters and a kernel size of 3
            conv2 = tf.layers.conv1d(conv1, 128, 5, activation=tf.nn.relu)
            # Max Pooling (down-sampling) with strides of 2 and kernel size of 2
            conv2 = tf.layers.max_pooling1d(conv2, 2, 2)
            conv2 = tf.layers.dropout(conv2, rate=dropout, training=is_training)

            # # Convolution Layer with 64 filters and a kernel size of 3
            conv3 = tf.layers.conv1d(conv2, 128, 5, activation=tf.nn.relu)
            # Max Pooling (down-sampling) with strides of 2 and kernel size of 2
            conv3 = tf.layers.max_pooling1d(conv3, 2, 2)
            conv3 = tf.layers.dropout(conv3, rate=dropout, training=is_training)
            
            
            # Convolution Layer with 64 filters and a kernel size of 3
            conv4 = tf.layers.conv1d(conv3, 128, 5, activation=tf.nn.relu)
            # Max Pooling (down-sampling) with strides of 2 and kernel size of 2
            conv4 = tf.layers.max_pooling1d(conv4, 2, 2)
            conv4 = tf.layers.dropout(conv4, rate=dropout, training=is_training)
            
            # Convolution Layer with 64 filters and a kernel size of 3
            conv5 = tf.layers.conv1d(conv4, 128, 5, activation=tf.nn.relu)
            # Max Pooling (down-sampling) with strides of 2 and kernel size of 2
            conv5 = tf.layers.max_pooling1d(conv5, 2, 2)
            conv5 = tf.layers.dropout(conv5, rate=dropout, training=is_training)
            
            # Convolution Layer with 64 filters and a kernel size of 3
            conv6 = tf.layers.conv1d(conv5, 128, 5, activation=tf.nn.relu)
            # Max Pooling (down-sampling) with strides of 2 and kernel size of 2
            conv6 = tf.layers.max_pooling1d(conv6, 2, 2)
            
            
            conv6 = tf.layers.average_pooling1d(conv6, 2, 2)
            
            # conv6 = tf.layers.dropout(conv6, rate=dropout, training=is_training)
            # Flatten the data to a 1-D vector for the fully connected layer
            fc1 = tf.contrib.layers.flatten(conv6)

            # Fully connected layer (in tf contrib folder for now)
            fc1 = tf.layers.dense(fc1, 256)
            # Apply Dropout (if is_training is False, dropout is not applied)
            fc1 = tf.layers.dropout(fc1, rate=dropout, training=is_training)
            
            fc2 = tf.layers.dense(fc1, 128)
            fc2 = tf.layers.dropout(fc2, rate=dropout, training=is_training)
            
            fc3 = tf.layers.dense(fc2, 64)
            fc3 = tf.layers.dropout(fc3, rate=dropout, training=is_training)

            # Output layer, class prediction
            out = tf.layers.dense(fc3, n_classes)

        return out


    # In[55]:


    # Define the model function (following TF Estimator Template)
    def model_fn(features, labels, mode):
        
        # Build the neural network
        # Because Dropout have different behavior at training and prediction time, we
        # need to create 2 distinct computation graphs that still share the same weights.
        logits_train = conv_net(features, num_classes, dropout, reuse=False, is_training=True)
        logits_test = conv_net(features, num_classes, dropout, reuse=True, is_training=False)
        
        # Predictions
        pred_classes = tf.argmax(logits_test, axis=1)
        pred_probas = tf.nn.softmax(logits_test)
        
        # If prediction mode, early return
        if mode == tf.estimator.ModeKeys.PREDICT:
            return tf.estimator.EstimatorSpec(mode, predictions=pred_classes) 
            
        # Define loss and optimizer
        loss_op = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=logits_train, labels=tf.cast(labels, dtype=tf.int32)))
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        train_op = optimizer.minimize(loss_op, global_step=tf.train.get_global_step())
        
        # Evaluate the accuracy of the model
        acc_op = tf.metrics.accuracy(labels=labels, predictions=pred_classes)
        
        # TF Estimators requires to return a EstimatorSpec, that specify
        # the different ops for training, evaluating, ...
        estim_specs = tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=pred_classes,
        loss=loss_op,
        train_op=train_op,
        eval_metric_ops={'accuracy': acc_op})

        return estim_specs


    # In[56]:


    #Build the Estimator
    model = tf.estimator.Estimator(model_fn)


    # In[57]:


    # from tensorflow.examples.tutorials.mnist import input_data
    import numpy as np
    # mnist = input_data.read_data_sets("/tmp/data/", one_hot=False)
    # Evaluate the Model
    # Define the input function for evaluating
    input_fn = tf.estimator.inputs.numpy_input_fn(
        x={'images': np.array(X_train)}, y=np.array(y_train),
        batch_size=batch_size, shuffle=False)
    # Use the Estimator 'evaluate' method
    model.evaluate(input_fn)


    # In[58]:


    input_fn = tf.estimator.inputs.numpy_input_fn(
        x={'images': np.array(X_test)}, shuffle=False)
    # Use the model to predict the images class
    preds = list(model.predict(input_fn))


    # In[59]:


    from sklearn.metrics import f1_score
    preds = np.array(preds)
    test_acc = f1_score(preds, np.array(y_test), average='micro')
    preds
    print(test_acc)


    # In[64]:


    #TEST ON TESTING DATASET RESERVED
    #predict now
    test_input = scaler.transform(test_input)
    input_fn = tf.estimator.inputs.numpy_input_fn(
        x={'images': np.array(test_input)}, shuffle=False)
    # Use the model to predict the images class
    test_input_pred = list(model.predict(input_fn))
    # test_input_pred = test_input_pred.squeeze()
    # test_input_pred.shape


    # In[65]:


    #write the output
    #interpolate
    #Scale data
    predicted_output = {'y': test_input_pred}
    predicted_output_df = pd.DataFrame(data=predicted_output)
    predicted_output_df.to_csv("../data/y_test.csv", index_label='id')


    # In[67]:


    len(test_input_pred)


    # In[63]:


    len(test_input_pred)


if __name__ == '__main__':
    tf.app.run()

