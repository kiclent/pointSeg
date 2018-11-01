


import sys
sys.path.append('../')

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2,3'
from lib.utils import *

train_path = '../inputs/npy/train/'
valid_path = '../inputs/npy/valid/'
test_path = '../inputs/npy/test2/'
results_path = './results2'

class_weights = [
    2700405826,
    4959969,
    2770858,
    127927068,
    47280570,
    1521167,
    190404,
    8409354
]
class_weights = np.array(class_weights, np.float32).reshape((1, 8))
class_weights = class_weights/class_weights.sum()
class_weights = 1/np.log(1.005+class_weights)
print(class_weights)


# 超参数
init_learning_rate = 1e-4
init_drop_rate = 0.2
total_epochs = 200000
batch_size = 8192
train_verbose = 100
valid_verbose = 2500
save_model_per_step = 5000
reduce_lr_per_step = -1

is_training = False

num_class = 8
num_features = 4
cnn_features = 7
flame_size = 58368

#  模型建立
import tensorflow as tf
from time import time as tc

def tf_coord_transform(tf_xyz, n_rows):
    with tf.name_scope(name='coord_transform') as scope:
        epsilon = 1e-6
        x = tf.slice(tf_xyz, [0, 0], [n_rows, 1])
        y = tf.slice(tf_xyz, [0, 1], [n_rows, 1])
        z = tf.slice(tf_xyz, [0, 2], [n_rows, 1])

        Rxyz = tf.sqrt(tf.reduce_sum(tf.square(tf.concat([x, y ,z], axis=1)), axis=1, keep_dims=True))
        Rxy = tf.sqrt(tf.reduce_sum(tf.square(tf.concat([x, y], axis=1)), axis=1, keep_dims=True))

        theta = x / (Rxy + epsilon)
        phi = z / (Rxyz + epsilon)
        tf_RTP = tf.concat((Rxyz, theta, phi), axis=1)
        return tf_RTP

training_flag = tf.placeholder(tf.bool, name='training_flag')
learning_rate = tf.placeholder(tf.float32, name='learning_rate')
dropout_rate = tf.placeholder(tf.float32, name='dropout_rate')
loss_weights = tf.placeholder(tf.float32, [1, num_class], name='loss_weights')

model_name = 'dcnn3_ball_coord'
checkpoint_dir = './model_{}/'.format(model_name)
log_dir = './logs/{}'.format(model_name)
ckpt_name = '{}.ckpt'.format(model_name)


tf_X = tf.placeholder(tf.float32, [flame_size, num_features], name='tf_X')
tf_Y = tf.placeholder(tf.float32, [flame_size, num_class], name='tf_Y')

tf_xyz = tf.slice(tf_X, [0, 0], [flame_size, 3])

tf_RTP = tf_coord_transform(tf_xyz, flame_size)

print(tf_xyz)
tf_cnn_inputs = tf.concat((tf_X, tf_RTP), axis=1, name='cnn_inputs')
tf_cnn_inputs = tf.reshape(tf_cnn_inputs, [1, flame_size, cnn_features, 1])

# ============================ 15 * 15 ===============================
conv1 = tf.layers.conv2d(inputs=tf_cnn_inputs,
                         filters=32,
                         kernel_size=(15, cnn_features),
                         strides=(8, cnn_features),
                         activation=tf.nn.relu,
                         padding='same',
                         name='conv1')
print('conv1:', conv1)

# output 1 x flame_size/4 x 1 x 1
max_pool1 = tf.layers.max_pooling2d(inputs=conv1,
                                    pool_size=(8, 1),
                                    strides=(4, 1),
                                    padding='same',
                                    name='max_pool1')
print('max_pool1:', max_pool1)

conv2 = tf.layers.conv2d(inputs=max_pool1,
                         filters=64,
                         kernel_size=(15, 1),
                         strides=(8, 1),
                         activation=tf.nn.relu,
                         padding='same',
                         name='conv2')
print('conv2:', conv2)

max_pool2 = tf.layers.max_pooling2d(inputs=conv2,
                                    pool_size=(8, 1),
                                    strides=(4, 1),
                                    padding='same',
                                    name='max_pool2')
print('max_pool2:', max_pool2)
print()


# ============================ 8 * 8 ===============================
conv3 = tf.layers.conv2d(inputs=tf_cnn_inputs,
                         filters=32,
                         kernel_size=(9, cnn_features),
                         strides=(4, cnn_features),
                         activation=tf.nn.relu,
                         padding='same',
                         name='conv3')
print('conv3:', conv3)

max_pool3 = tf.layers.max_pooling2d(inputs=conv3,
                                    pool_size=(4, 1),
                                    strides=(2, 1),
                                    padding='valid',
                                    name='max_pool3')
print('max_pool3:', max_pool3)

conv4 = tf.layers.conv2d(inputs=max_pool3,
                         filters=64,
                         kernel_size=(9, 1),
                         strides=(4, 1),
                         activation=tf.nn.relu,
                         padding='same',
                         name='conv4')
print('conv4:', conv4)

max_pool4 = tf.layers.max_pooling2d(inputs=conv4,
                                    pool_size=(4, 1),
                                    strides=(4, 1),
                                    padding='valid',
                                    name='max_pool4')

print('max_pool4:', max_pool4)

# ============================ tile ===============================
tile = tf.tile(max_pool2, [1, 1, 8, 1])
tile = tf.reshape(tile, [1, 456, 1, 64])
print('tile:', tile)

conv_merged = tf.layers.conv2d(inputs=tf.concat([tile, max_pool4], axis=3),
                               filters=64,
                               kernel_size=(3, 1),
                               strides=(1, 1),
                               activation=tf.nn.relu,
                               padding='same',
                               name='conv_merged')

print('conv_merged:', conv_merged)

deconv1 = tf.layers.conv2d_transpose(conv_merged, 64, (8, 1), (4, 1), activation=tf.nn.relu, padding='same', name='deconv1')
print('deconv1:', deconv1)

deconv2 = tf.layers.conv2d_transpose(deconv1, 64, (8, 1), (4, 1), activation=tf.nn.relu, padding='same', name='deconv2')
print('deconv2:', deconv2)

deconv3 = tf.layers.conv2d_transpose(deconv2, 32, (8, 1), (4, 1), activation=tf.nn.relu, padding='same', name='deconv3')
print('deconv3:', deconv3)

deconv4 = tf.layers.conv2d_transpose(deconv3, 16, (8, 1), (2, 1), activation=tf.nn.relu, padding='same', name='deconv4')
print('deconv4:', deconv4)

deconv4 = tf.reshape(deconv4, [flame_size, 16])

# 建立一个全连接网络

fc = tf.layers.dense(inputs=tf.concat([tf.reshape(tf_cnn_inputs, [-1, cnn_features]), deconv4], axis=1),
                     units=64,
                     activation=tf.nn.relu,
                     name='hidden_fc')
print('fc:', fc)

logits = tf.layers.dense(tf.concat((tf.reshape(tf_cnn_inputs, [-1, cnn_features]), fc), axis=1), num_class, activation=None)
print('logits:', logits)

probability = tf.nn.softmax(logits, name='probability')
print('probability:', probability)

# 加权 loss
entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=tf_Y)

error = tf.nn.weighted_cross_entropy_with_logits(targets=tf_Y, logits=logits, pos_weight=loss_weights)
print('entropy:', entropy)
print('error:', error)


loss = tf.reduce_mean(error)

correct_prediction = tf.cast(tf.equal(tf.argmax(logits, 1), tf.argmax(tf_Y, 1)), tf.float32)
accuracy = tf.reduce_mean(correct_prediction)

optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)

tf.summary.scalar('loss', loss)
tf.summary.scalar('accuracy', accuracy)

init = tf.global_variables_initializer()

saver = tf.train.Saver(tf.global_variables(), max_to_keep=20)

# ==================================================
# 训练
with tf.Session() as sess:
    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)

    sess.run(init)

    if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
        print('loading model {}'.format(ckpt.model_checkpoint_path))
        saver.restore(sess, ckpt.model_checkpoint_path)
        global_step = int(ckpt.model_checkpoint_path.split('-')[-1]) + 1
    else:
        global_step = 0

    if is_training:
        merged = tf.summary.merge_all()
        summary_writer_train = tf.summary.FileWriter(log_dir + '/train', sess.graph)
        summary_writer_valid = tf.summary.FileWriter(log_dir + '/valid')

        tic = tc()
        epoch_learning_rate = init_learning_rate

        for epoch in range(total_epochs):
            local_step = 0
            train_idxs = np.random.permutation(499)
            for i in range(499):
                print('flame: {}/{} '.format(i+1, 499))
                data, _ = npy_item_read(train_path, train_idxs[i])
                flame_idxs = np.unique(data[:, 0])
                np.random.shuffle(flame_idxs)
                for flame in flame_idxs:

                    # 处理每一帧数据
                    lg = data[:, 0] == flame
                    pts, intensity, category = data[lg, 1:4], data[lg, 4], data[lg, 5]

                    # 对每一帧数据长度进行补齐
                    pts, intensity, category = flame_len_fix(pts, intensity, category, flame_size)

                    # 归一化
                    pts[:, 0], pts[:, 1], pts[:, 2] = pts[:, 0]/263, pts[:, 1]/219, pts[:, 2]/90
                    intensity = intensity*2 - 1

                    #
                    X = np.hstack((pts, intensity))
                    Y = np.zeros((category.shape[0], num_class), np.float32)

                    for i in range(num_class):
                        Y[:, i] = (category == i).reshape(-1)
                        # print(i, Y[:, i].sum())

                    local_step += 1
                    global_step += 1

                    # reduce learning rate
                    if (reduce_lr_per_step > 0) and (epoch_learning_rate > 1e-5) and (global_step % reduce_lr_per_step == 0):
                        epoch_learning_rate /= 3
                        print('reduce learning rate to :', epoch_learning_rate)

                    if global_step % valid_verbose == 0:

                        data_valid, _ = npy_item_read(valid_path, 499)
                        valid_probs = []
                        valid_labels = []
                        valid_loss = []
                        for flame_valid in np.unique(data_valid[:, 0]):

                            # 处理每一帧数据
                            lg = data_valid[:, 0] == flame_valid
                            pts, intensity, category = data_valid[lg, 1:4], data_valid[lg, 4], data_valid[lg, 5]

                            # 对每一帧数据长度进行补齐
                            pts, intensity, category = flame_len_fix(pts, intensity, category, flame_size)

                            # 归一化
                            pts[:, 0], pts[:, 1], pts[:, 2] = pts[:, 0] / 263, pts[:, 1] / 219, pts[:, 2] / 90
                            intensity = intensity * 2 - 1

                            #
                            x_valid = np.hstack((pts, intensity))
                            y_valid = np.zeros((category.shape[0], num_class), np.float32)

                            for i in range(num_class):
                                y_valid[:, i] = (category == i).reshape(-1)

                            test_feed_dict = {
                                tf_X: x_valid,
                                tf_Y: y_valid,
                                learning_rate: epoch_learning_rate,
                                dropout_rate: 0,
                                loss_weights: class_weights,
                                training_flag: False
                            }

                            _probs, _loss, valid_acc, valid_summary = sess.run(
                                [probability, loss, accuracy, merged],
                                feed_dict=test_feed_dict)
                            summary_writer_valid.add_summary(valid_summary, global_step=global_step)
                            valid_probs.append(_probs)
                            valid_labels.append(y_valid)
                            valid_loss.append(_loss)
                        valid_probs = np.vstack(valid_probs)
                        valid_labels = np.vstack(valid_labels)
                        iou_, iou = get_iou(valid_labels, prob2hot(valid_probs))
                        TP = (prob2hot(valid_probs) == 1) * valid_labels
                        acc_by_class = TP.sum(axis=0)/valid_labels.sum(axis=0)
                        acc = (TP.sum(axis=1)>0).mean()
                        print('======================== valid ========================')
                        print(
                            'Epoch:{:03d}, '.format(epoch + 1),
                            'Mean loss:{:.3f}, '.format(np.mean(valid_loss)),
                            'acc:{:.5f}, '.format(acc),
                            'iou:{:.5f}, '.format(iou),
                            'epoch_lr {:.7f}, '.format(epoch_learning_rate),
                            '{:.2f} s.'.format(tc() - tic)
                        )

                        acc_by_class = ' '.join(['{:.3f}'.format(s) for s in acc_by_class])
                        print('acc_by_class: ', acc_by_class)

                        iou_ = ' '.join(['{:.3f}'.format(s) for s in iou_])
                        print('iou: ', iou_)
                        print('========================================================\n')

                    train_feed_dict = {
                        tf_X: X,
                        tf_Y: Y,
                        learning_rate: epoch_learning_rate,
                        dropout_rate: init_drop_rate,
                        loss_weights: class_weights, # loss_weights
                        training_flag: True
                    }

                    _, probs, train_loss, train_acc, train_summary = sess.run([optimizer, probability, loss, accuracy, merged],
                                                                       feed_dict=train_feed_dict)

                    if global_step % train_verbose == 0:
                        TP = (prob2hot(probs) * Y).sum(axis=0)
                        acc_by_class = TP/((prob2hot(probs) + Y)>0).sum(axis=0)
                        acc = (TP > 0).mean()
                        print('---------------------------------------------------')
                        print(
                            'step:{:05d}, '.format(global_step),
                            'train_loss:{:.7f}'.format(train_loss),
                            'acc:{:.7f}, '.format(acc),
                            '{:.2f} s.'.format(tc() - tic)
                        )
                        print()
                        acc_by_class = ' '.join(['{:.3f}'.format(s) for s in acc_by_class])
                        print('AP_by_class: ', acc_by_class)
                        print('---------------------------------------------------\n')
                        summary_writer_train.add_summary(train_summary, global_step=global_step)

                    if global_step % save_model_per_step == 0:
                        print('saving model {}-{}'.format(ckpt_name, global_step))
                        saver.save(sess, checkpoint_dir + ckpt_name, global_step)

                    # --------------------------
                    # print valid profile
        saver.save(sess, checkpoint_dir + ckpt_name, global_step=global_step)
    else:

        if not os.path.exists(results_path):
            os.mkdir(results_path)
        tic = tc()
        for batch_data, name_index in npy_read(test_path, is_training=False):

            pts = batch_data[:, 1:4]
            intensity = batch_data[:, 4].reshape((-1, 1))
            print('data shape:', batch_data.shape)
            print('read {:.4f} s.'.format(tc() - tic))
            # 数据处理

            for flame in np.unique(batch_data[:, 0]):

                # 处理每一帧数据
                lg = batch_data[:, 0] == flame
                pts, intensity = batch_data[lg, 1:4], batch_data[lg, 4]

                # 对每一帧数据长度进行补齐
                pts, intensity, fix_head, fix_tail = flame_len_fix_test(pts, intensity, flame_size)

                # 归一化
                pts[:, 0], pts[:, 1], pts[:, 2] = pts[:, 0]/263, pts[:, 1]/219, pts[:, 2]/90
                intensity = intensity*2 - 1

                #
                X = np.hstack((pts, intensity))

                #print('transform {:.4f} s.'.format(tc() - tic))
                test_feed_dict = {
                    tf_X: X,
                    learning_rate: 0.1,
                    dropout_rate: 0,
                    loss_weights: class_weights,
                    training_flag: False
                }
                probs = sess.run(probability, feed_dict=test_feed_dict)
                preds = np.argmax(probs, axis=1).reshape(-1)

                pd.DataFrame(preds[fix_head:-fix_tail]).to_csv(os.path.join(results_path, name_index.iloc[:, 0][flame]), header=None, index=False)

            print('write {:.4f} s.'.format(tc() - tic))
            print('\n---------------------------')














