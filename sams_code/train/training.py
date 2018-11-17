import tensorflow as tf
import sys
import os.path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.metrics import f1_score
from io import BytesIO
from tensorflow.python.lib.io import file_io




sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
#tf.set_random_seed(0)
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('data_dir', 'data', 'Input Directory.')
flags.DEFINE_string('output_dir', 'output', 'Output Directory.')
flags.DEFINE_string('experiment_name', 'conditional_model_1_clipping',
                    'Experiment Name.')


class Model(object):
    def __init__(self, is_training=True):
        # define args here for now
        #model variables to be accessed from trianing
        # accessible variables for training:
        self.learning_rate = 0.001
        self.x = None
        self.y = None
        self.loss = None
        self.acc_op = None
        self.summary_ops = None
        self.dropout = 0.25
        self.is_training = None
        self.pred_classes = None


    def build_model(self):
        global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')
        self.global_step = tf.assign(global_step, global_step+1)
        with tf.name_scope('data'):
            self.is_training = tf.placeholder(tf.bool,
                                name='input')
            self.x = tf.placeholder(tf.float32,
                               shape=[None, 18154],
                               name='input')
            new_x = tf.expand_dims(self.x, -1)
            self.y = tf.placeholder(tf.float32,
                                    shape=[None],
                                    name='target')
            print(self.y)
        with tf.name_scope('model'):
            conv1 = tf.layers.conv1d(new_x, 128, 5, activation=tf.nn.relu)
            conv1 = tf.layers.max_pooling1d(conv1, 2, 2)
            conv1 = tf.layers.dropout(conv1, rate=self.dropout, training=self.is_training)

            conv2 = tf.layers.conv1d(conv1, 128, 5, activation=tf.nn.relu)
            conv2 = tf.layers.max_pooling1d(conv2, 2, 2)
            conv2 = tf.layers.dropout(conv2, rate=self.dropout, training=self.is_training)

            conv3 = tf.layers.conv1d(conv2, 128, 5, activation=tf.nn.relu)
            conv3 = tf.layers.max_pooling1d(conv3, 2, 2)
            conv3 = tf.layers.dropout(conv3, rate=self.dropout, training=self.is_training)
            
            
            conv4 = tf.layers.conv1d(conv3, 128, 5, activation=tf.nn.relu)
            conv4 = tf.layers.max_pooling1d(conv4, 2, 2)
            conv4 = tf.layers.dropout(conv4, rate=self.dropout, training=self.is_training)
            
            conv5 = tf.layers.conv1d(conv4, 128, 5, activation=tf.nn.relu)
            conv5 = tf.layers.max_pooling1d(conv5, 2, 2)
            conv5 = tf.layers.dropout(conv5, rate=self.dropout, training=self.is_training)
            
            conv6 = tf.layers.conv1d(conv5, 128, 5, activation=tf.nn.relu)
            conv6 = tf.layers.max_pooling1d(conv6, 2, 2)
            
            
            conv6 = tf.layers.average_pooling1d(conv6, 2, 2)
            
            fc1 = tf.contrib.layers.flatten(conv6)

            fc1 = tf.layers.dense(fc1, 256)
            fc1 = tf.layers.dropout(fc1, rate=self.dropout, training=self.is_training)
            
            fc2 = tf.layers.dense(fc1, 128)
            fc2 = tf.layers.dropout(fc2, rate=self.dropout, training=self.is_training)
            
            fc3 = tf.layers.dense(fc2, 64)
            fc3 = tf.layers.dropout(fc3, rate=self.dropout, training=self.is_training)

            out = tf.layers.dense(fc3, 4)
            self.pred_classes = tf.argmax(out, axis=1)
            pred_probas = tf.nn.softmax(out)

        with tf.name_scope('network_loss'):
            print(out)
            print(self.y)
            self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=out, labels=tf.cast(self.y, dtype=tf.int32)))
            optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
            self.train_op = optimizer.minimize(self.loss, global_step=tf.train.get_global_step())
            self.acc_op = tf.metrics.accuracy(labels=self.y, predictions=self.pred_classes)          
  
        with tf.name_scope("summaries"):
            tf.summary.scalar('loss', self.loss)
            tf.summary.scalar('acc_op', tf.reduce_mean(self.acc_op))
            self.summary_ops = tf.summary.merge_all()
        return

class TrainModel(object):
    def __init__(self):
        # self.train_input = [1, 1, 1, 1]
        # self.train_output = [2]
        # self.test_input = [1, 1, 1, 1]
        train_input = pd.read_csv(BytesIO(file_io.read_file_to_string(os.path.join(FLAGS.data_dir, 'X_train.csv'), binary_mode=True)))
        train_output = pd.read_csv(BytesIO(file_io.read_file_to_string(os.path.join(FLAGS.data_dir, 'y_train.csv'), binary_mode=True)))
        test_input = pd.read_csv(BytesIO(file_io.read_file_to_string(os.path.join(FLAGS.data_dir, 'X_test.csv'), binary_mode=True)))
        # m = train_input.head(10)
        # k = train_output.head(10)
        # c = test_input.head(10)
        # m.to_csv(os.path.join(FLAGS.data_dir, 'X_train.csv'))
        # k.to_csv(os.path.join(FLAGS.data_dir, 'y_train.csv'))
        # c.to_csv(os.path.join(FLAGS.data_dir, 'X_test.csv'))
        #PREPROCESSING
        train_input = train_input.sort_values(by=['id'])
        train_input = train_input.drop(columns=['id'])
        train_output = train_output.sort_values(by=['id'])
        train_output = train_output.drop(columns=['id'])
        test_input = test_input.sort_values(by=['id'])
        test_input = test_input.drop(columns=['id'])

        # test_input.apply(lambda x: x.loc[x.last_valid_index():] = x.loc[x.last_valid_index() - (9000 - x.last_valid_index() - 1):x.last_valid_index() + 1], axis=1)
        train_input = train_input.interpolate(method='linear', axis=1)
        test_input = test_input.interpolate(method='linear', axis=1)

        train_output = train_output.squeeze()
        X_train, X_test, self.y_train, self.y_test = train_test_split(train_input, train_output, test_size=0.2)
        scaler = StandardScaler()
        scaler.fit(X_train)
        self.X_train = np.array(scaler.transform(X_train))
        self.X_test = np.array(scaler.transform(X_test))
        self.test_input = np.array(scaler.transform(test_input))
        self.y_train = np.array(self.y_train)
        self.y_test = np.array(self.y_test)
        #params
        self.epochs = 200

    def shuffle_batches(self, inputs, targets, batchsize):
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
        for n in range(0, len(inputs)-batchsize+1, batchsize):
            batch = indices[n:n+batchsize]
            batch_targets = targets[batch]
            # batch_targets = np.array(targets[batch]).reshape(-1)
            # print(batch_targets)
            # batch_targets = np.eye(4)[batch_targets]
            # print(np.shape(batch_targets))
            yield inputs[batch], batch_targets

    def train(self):
        self.train_model = Model() # to be changed
        batchsize = 64
        self.train_model.build_model()
        saver = tf.train.Saver(max_to_keep=10)
        summary_proto = tf.Summary()
        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
            init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
            sess.run(init)
            writer_file_path = os.path.join(FLAGS.output_dir, FLAGS.experiment_name, 'improved_graph')
            checkpoint_file = os.path.join(FLAGS.output_dir, FLAGS.experiment_name, 'checkpoints')
            writer = tf.summary.FileWriter(writer_file_path, sess.graph)
            for epoch in range(0, self.epochs):
                print("Epoch number " + str(epoch))
                batch_idx = 0
                training_loss = 0.0
                for batch in self.shuffle_batches(self.X_train, self.y_train, batchsize):
                    inputs, targets = batch
                    feed_dict = {self.train_model.x: inputs,
                                 self.train_model.y: targets,
                                 self.train_model.is_training: True}
                    global_step, summary_train, accuracy, network_loss, _ = sess.run([self.train_model.global_step,
                                                                                   self.train_model.summary_ops,
                                                                                   self.train_model.acc_op,
                                                                                   self.train_model.loss,
                                                                                   self.train_model.train_op],
                                                                                   feed_dict=feed_dict)
                    training_loss += network_loss
                    batch_idx += 1
                    writer.add_summary(summary_train, global_step=global_step)
                    if batch_idx % 1 == 0:
                        print('Epoch ', epoch, ' and Batch ', batch_idx, ' | training loss is ',
                              training_loss / batch_idx)
                    # if batch_idx % 10 == 0:
                    #     saver.save(sess, checkpoint_file, global_step=global_step)
                    #     summary_proto.ParseFromString(summary_train)
                num_of_training_batches = batch_idx
                validation_loss = 0.
                batch_idx = 0
                #VALIDATION
                
                validation_feed = {self.train_model.x: self.X_test,
                                   self.train_model.y: self.y_test,
                                   self.train_model.is_training: False}
                [predicted_classes] = sess.run([self.train_model.pred_classes],
                                                          feed_dict=validation_feed)
                predicted_classes = predicted_classes
                print(predicted_classes)
                test_acc = f1_score(predicted_classes, np.array(self.y_test), average='micro')
                print('Epoch ', epoch, ' got score of  ', test_acc)

                #FINAL TESTINg
            testing_feed = {self.train_model.x: self.test_input,
                               self.train_model.is_training: False}
            [predicted_classes] = sess.run([self.train_model.pred_classes],
                                                      feed_dict=testing_feed)
            test_input_pred = predicted_classes + 1
            predicted_output = {'y': test_input_pred}
            predicted_output_df = pd.DataFrame(data=predicted_output)
            predicted_output_df.to_csv(os.path.join(FLAGS.output_dir, 'y_test.csv'), index_label='id')


def main(_):
    training = TrainModel()
    training.train()


if __name__ == '__main__':
    tf.app.run()


