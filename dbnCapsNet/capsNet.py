import tensorflow as tf

from capsLayer import CapsLayer
from config import cfg
import numpy as np
import utils
import os
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import time

epsilon = 1e-9


class CapsNet(object):

    def __init__(self, is_training=True, dbn=None):
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.X = tf.placeholder(tf.float32, shape=(cfg.batch_size, cfg.input_size))
            self.Y = tf.placeholder(tf.int32, shape=(cfg.batch_size, cfg.out_size))
            if is_training:
                # nn
                with tf.variable_scope('nn'):
                    self.nn_w = [tf.Variable(dbn.rbm_list[i].w) for i in range(cfg.nn_layer_size)]
                    self.nn_b = [tf.Variable(dbn.rbm_list[i].hb) for i in range(cfg.nn_layer_size)]

                self.build_arch()
                self.loss()

                # t_vars = tf.trainable_variables()
                self.global_step = tf.Variable(0, name='global_step', trainable=False)
                self.learning_rate = tf.train.exponential_decay(cfg.caps_startLr, self.global_step,
                                                                cfg.caps_decay_steps,
                                                                cfg.caps_decay_rate)
                self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
                self.train_op = self.optimizer.minimize(self.margin_loss,
                                                        global_step=self.global_step)  # var_list=t_vars)
                self._summary()

            else:
                with tf.variable_scope('nn'):
                    self.nn_w = []
                    self.nn_b = []
                    # for i in range(cfg.nn_layer_size):
                    #     if i == 0:
                    #         self.nn_w.append(tf.Variable(tf.random_normal([cfg.input_size, cfg.nn_hsizes[i]])))
                    #     else:
                    #         self.nn_w.append(tf.Variable(tf.random_normal([cfg.nn_hsizes[1 - 1], cfg.nn_hsizes[i]])))
                    #     self.nn_b.append(tf.Variable(tf.random_normal([cfg.nn_hsizes[i]])))
                    self.nn_w = [tf.Variable(dbn.rbm_list[i].w) for i in range(cfg.nn_layer_size)]
                    self.nn_b = [tf.Variable(dbn.rbm_list[i].hb) for i in range(cfg.nn_layer_size)]
                self.build_arch()


            self.saver = tf.train.Saver()
            self.sess = tf.Session()
            self.init = tf.global_variables_initializer()
            if not is_training:
                self.saver.restore(self.sess, './model_path/test_model.ckpt')

    def build_arch(self):
        with tf.variable_scope('nn'):
            # Conv1, [batch_size, 20, 20, 256]
            nn_out = [self.X]
            for i in range(cfg.nn_layer_size):
                nn_out.append(cfg.nn_activate(tf.matmul(nn_out[-1], self.nn_w[i]) + self.nn_b[i]))

        # DigitCaps layer, return [batch_size, 10, 16, 1]
        with tf.variable_scope('PrimaryCaps_layer'):
            primaryCaps = CapsLayer(num_outputs=cfg.primaryCaps_out_num,
                                    vec_len=cfg.primaryCaps_vec_num, with_routing=False, layer_type='NN')
            caps1 = primaryCaps(nn_out[-1])

        # DigitCaps layer, return [batch_size, 10, 16, 1]
        with tf.variable_scope('DigitCaps_layer'):
            digitCaps = CapsLayer(num_outputs=cfg.out_size,
                                  vec_len=cfg.outCaps_vec_num, with_routing=True, layer_type='FC')
            self.caps2 = digitCaps(caps1)

        # Decoder structure in Fig. 2
        # 1. Do masking, how:
        with tf.variable_scope('Masking'):
            # a). calc ||v_c||, then do softmax(||v_c||)
            # [batch_size, 10, 16, 1] => [batch_size, 10, 1, 1]
            self.v_length = tf.sqrt(tf.reduce_sum(tf.square(self.caps2),
                                                  axis=2, keep_dims=True) + epsilon)

            self.v_length = tf.reshape(self.v_length, [-1, 2])
            # b). pick out the index of max softmax val of the 10 caps
            # [batch_size, 10, 1, 1] => [batch_size] (index)
            self.argmax_idx = tf.to_int32(tf.argmax(self.v_length, axis=1))
            # self.argmax_idx = tf.contrib.layers.flatten(self.argmax_idx)


    def loss(self):
        # 1. The margin loss

        # [batch_size, 10, 1, 1]
        # max_l = max(0, m_plus-||v_c||)^2
        max_l = tf.square(tf.maximum(0., cfg.m_plus - self.v_length))
        # max_r = max(0, ||v_c||-m_minus)^2
        max_r = tf.square(tf.maximum(0., self.v_length - cfg.m_minus))

        # reshape: [batch_size, 10, 1, 1] => [batch_size, 10]
        max_l = tf.reshape(max_l, shape=(cfg.batch_size, -1))
        max_r = tf.reshape(max_r, shape=(cfg.batch_size, -1))

        # calc T_c: [batch_size, 10]
        # T_c = Y, is my understanding correct? Try it.
        T_c = tf.cast(self.Y, tf.float32)
        # [batch_size, 10], element-wise multiply
        L_c = T_c * max_l + cfg.lambda_val * (1 - T_c) * max_r

        self.margin_loss = tf.reduce_mean(tf.reduce_sum(L_c, axis=1))



    # Summary
    def _summary(self):
        train_summary = []
        train_summary.append(tf.summary.scalar('train/margin_loss', self.margin_loss))
        train_summary.append(tf.summary.scalar('train/learning_rate', self.learning_rate))
        self.train_summary = tf.summary.merge(train_summary)

        correct_prediction = tf.equal(tf.to_int32(tf.argmax(self.Y, axis=-1)), self.argmax_idx)
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    def train(self, trX, trY, vaX, vaY, train_writer_path, log_f):
        sess = self.sess
        saver = self.saver
        if train_writer_path:
            train_writer = tf.summary.FileWriter(train_writer_path, sess.graph)
        time.clock()
        sess.run(self.init)
        _global_step = 0
        epoch = 0
        inds = [i for i in range(len(trX))]
        val_acc = 0.
        tr_acc = 0.
        acc_plot = []
        tr_acc_plot = []
        # try:
        while epoch < cfg.caps_epochs:
            epoch += 1
            np.random.shuffle(inds)
            count = 0
            acc_= []
            for start, end in zip(range(0, len(trX)+1, cfg.batch_size), range(cfg.batch_size, len(trX)+1, cfg.batch_size)):

                _trX = trX[inds[start:end]]
                _trY = trY[inds[start:end]]
                _,  _margin_loss,  _global_step, _learning_rate, summary, acc = \
                    sess.run([self.train_op,  self.margin_loss,
                               self.global_step, self.learning_rate
                              , self.train_summary,self.accuracy
                              ],
                         feed_dict={self.X : _trX, self.Y : _trY})

                acc_.append(acc)
                count += 1
                if count % cfg.print_frq:
                    # utils.print_out('epoch %d, step %d, gloSet %d, lr %.4f,  margin_loss %.4f, acc %.4f'
                    #                 %(epoch, count, _global_step, _learning_rate ,  _margin_loss, acc))
                    #                 #, log_f)
                    if train_writer_path:
                        train_writer.add_summary(summary, global_step=_global_step)
                acc = np.sum(acc_)/count
            if epoch % cfg.val_frq:
                tr_acc_plot.append(acc)
                val_acc = 0.
                val_count = 0
                val_margin_loss = 0.

                for start, end in zip(range(0, len(vaX)+1, cfg.batch_size),
                                      range(cfg.batch_size, len(vaX)+1, cfg.batch_size)):
                    _vaX = vaX[start:end]
                    _vaY = vaY[start:end]
                    i_acc, i_margin_loss = sess.run([self.accuracy, self.margin_loss],
                                                   feed_dict={self.X: _vaX, self.Y: _vaY})

                    val_count += 1
                    val_acc += i_acc
                    val_margin_loss += i_margin_loss
                val_acc /= val_count
                val_margin_loss /= val_count
                acc_plot.append(val_acc)

                # tr_acc = 0.
                # tr_count = 0
                # tr_margin_loss = 0.
                #
                # for start, end in zip(range(0, len(trX), cfg.batch_size),
                #                       range(cfg.batch_size, len(trX), cfg.batch_size)):
                #     _trX = trX[start:end]
                #     _trY = trY[start:end]
                #     i_acc, i_margin_loss = sess.run([self.accuracy, self.margin_loss],
                #                                    feed_dict={self.X: _trX, self.Y: _trY})
                #     tr_count += 1
                #     tr_acc += i_acc
                #     tr_margin_loss += i_margin_loss
                # tr_acc /= tr_count
                # tr_margin_loss /= tr_count

                utils.print_out(
                    'epoch %d, global step %d,'
                    'training  acc %.4f, tr_margin_loss %.6f' % (
                    epoch, _global_step,  val_acc, val_margin_loss)
                    , log_f)
                # print('------------------')
                # print(max_id)
                # print(np.argmax(_vaY, axis=1))
                # print(tr_margin_loss, np.sum(np.equal(max_id, np.argmax(_vaY, axis=1)))/cfg.batch_size)
     #可视化数据
        # x = len(acc_plot)
        # y = acc_plot
        # y2 = tr_acc_plot
        # plt.figure()
        # plt.plot(range(0,x), y)
        # plt.plot(range(0,x), y2)
        # plt.show()
        ckpt_path = './model_path/test_model.ckpt'
        save_path = saver.save(sess, ckpt_path)


        # except (KeyboardInterrupt, InterruptedError) as e:
        #     utils.print_out(
        #         "!!!!  Interrupt ## end training, global step %d" % (
        #             _global_step), log_f)
        #     utils.print_out("An error occurred. {}".format(e.args[-1]))

        # finally:
        if train_writer_path:
            checkPoint_path = os.path.join(train_writer_path, "checkPoint.model")
            saver.save(sess, checkPoint_path, global_step=_global_step)
            utils.print_out(
            "   end training, global step %d, check point save to %s-%d" % (
                _global_step, checkPoint_path, _global_step), log_f)
        # sess.close()
        return val_acc, tr_acc
    def predict(self, pred_X):
        sess = self.sess
        pred_y = []
        for start, end in zip(range(0, len(pred_X) + cfg.batch_size - 1, cfg.batch_size),
                              range(cfg.batch_size, len(pred_X) + cfg.batch_size - 1, cfg.batch_size)):
            _pred_X = pred_X[start:end]
            _pred_X_len = len(_pred_X)
            if _pred_X_len <cfg.batch_size:
                # _pred_X = np.pad(((0, cfg.batch_size - _pred_X_len),(0,0)), 'constant', ((0,0), (0,0)))
                _pred_X = np.pad(_pred_X, ((0, cfg.batch_size - _pred_X_len), (0, 0)), 'constant')
            _pred_y = sess.run([self.argmax_idx], feed_dict={self.X: _pred_X})
            for i in range(_pred_X_len):
                pred_y.append(_pred_y[0][i])
        return np.asarray(pred_y, np.int32)
