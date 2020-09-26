import argparse
import numpy as np
import tensorflow as tf
import local_point_clouds
from pointnet_cls import *

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--log_dir', default='log', help='Log dir [default: log]')
parser.add_argument('--max_epoch', type=int, default=50, help='Epoch to run [default: 50]')
parser.add_argument('--batch_size', type=int, default=24, help='Batch Size during training [default: 24]')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Initial learning rate [default: 0.001]')
parser.add_argument('--momentum', type=float, default=0.9, help='Initial learning rate [default: 0.9]')
parser.add_argument('--optimizer', default='adam', help='adam or momentum [default: adam]')
parser.add_argument('--decay_step', type=int, default=300000, help='Decay step for lr decay [default: 300000]')
parser.add_argument('--decay_rate', type=float, default=0.5, help='Decay rate for lr decay [default: 0.5]')
parser.add_argument('--num_proteins',type=int,default=1000,help='number of proteins from db [default: 1000]')
FLAGS = parser.parse_args()

BATCH_SIZE = FLAGS.batch_size
MAX_EPOCH = FLAGS.max_epoch
BASE_LEARNING_RATE = FLAGS.learning_rate
GPU_INDEX = FLAGS.gpu
MOMENTUM = FLAGS.momentum
OPTIMIZER = FLAGS.optimizer
DECAY_STEP = FLAGS.decay_step
DECAY_RATE = FLAGS.decay_rate
NUM_PROTEINS = FLAGS.num_proteins
LOG_DIR = FLAGS.log_dir

if not os.path.exists(LOG_DIR): os.mkdir(LOG_DIR)
LOG_FOUT = open(os.path.join(LOG_DIR, 'log_train.txt'), 'w')
LOG_FOUT.write(str(FLAGS) + '\n')

CLOUD_SIZE = 8
NUM_CLASSES = 3  ### need to fit for our caser
new_label_dictionary = {
    0: 0, 1: 0, 2: 0,  # helix
    3: 1, 4: 1,  # sheet
    5: 2, 6: 2, 7: 2  # coil
}

BN_INIT_DECAY = 0.5
BN_DECAY_DECAY_RATE = 0.5
BN_DECAY_DECAY_STEP = float(DECAY_STEP)
BN_DECAY_CLIP = 0.99


# Load protein data
data_batch_list = []
label_batch_list = []
protein_info_dir = "./Proteins_Info"
for subdir_info, dirs_info, files_info in os.walk(protein_info_dir):
    for protein_dir in dirs_info:
        dir_name = os.path.join(subdir_info, protein_dir)
        for subdir, dirs, files in os.walk(dir_name):
            for protein_files in files:
                protein_name_npy = os.path.join(subdir, protein_files)
                if ("data" in protein_name_npy):
                    data_batch = np.load(protein_name_npy)
                elif ("label" in protein_name_npy):
                    label_batch = np.load(protein_name_npy)
                    for i, label in enumerate(label_batch):
                        label_batch[i] = new_label_dictionary[label]
                else:
                    print("ERROR: unknown file: " + protein_name_npy)
            if (data_batch.shape[0] > 0 and label_batch.shape[0] > 0):
                data_batch_list.append(data_batch)
                label_batch_list.append(label_batch)

data_batches = np.array(data_batch_list)
label_batches = np.array(label_batch_list)
# JUST FOR CHECK  - DELETE THE FOLLOING 2 lines
if(data_batches.shape[0] > NUM_PROTEINS ):
    data_batches = data_batches[0:NUM_PROTEINS]
    label_batches = label_batches[0:NUM_PROTEINS]
# print(data_batches.shape) #NUM_PROTEINS X num_point_in_protein X 3
# print(label_batches.shape) #NUM_PROTEINS X num_point_in_protein

NUM_PROTEINS=data_batches.shape[0]
# Centerelizing each proteing to (0,0,0)
for i in range(NUM_PROTEINS):
    xyz_mean = np.mean(data_batches[i], axis=0)[0:3]
    data_batches[i][:, 0:3] -= xyz_mean
    # print("MEAN i = ", i, " is ", np.mean(data_batches[i], axis=0))

# In the local model, we create a point-cloud of size 32 around each residue.
# That means that if we had:        N protines X 512 points X 3 coordinates,
# we now have:                      N protines * 512 points X 32 points on cloud X 3 coordinates.
local_data = []
local_label = []
for i in range(NUM_PROTEINS):
    local_cloud, _ = local_point_clouds.build_local_point_cloud(data_batches[i], CLOUD_SIZE)
    num_points_in_protein = data_batches[i].shape[0]
    # neigbors is of shape 512 X 32.
    for j in range(num_points_in_protein):
        local_data.append(local_cloud[j])
        local_label.append(label_batches[i][j])

local_data = np.array(local_data)
local_label = np.array(local_label)
print(local_data.shape)
print(local_label.shape)

assert (local_data.shape[0] == local_label.shape[0])
total_number_of_points = local_data.shape[0]

# division of proteins to test and train indexs
train_idxs = []
test_idxs = []
threshold_idx = (total_number_of_points * 5) // 6  # we take the 5/6 of the total number of proteins, and the rest to test
train_idxs = range(threshold_idx)
test_idxs = range(threshold_idx, total_number_of_points)
train_data = local_data[train_idxs, ...]
train_label = local_label[train_idxs]
test_data = local_data[test_idxs, ...]
test_label = local_label[test_idxs]
print(train_data.shape, train_label.shape, test_data.shape, test_label.shape)


def log_string(out_str):
    LOG_FOUT.write(out_str + '\n')
    LOG_FOUT.flush()
    print(out_str)


def shuffle_data(data, labels):
    """ Shuffle data and labels.
        Input:
          data: B,N,... numpy array
          label: B,... numpy array
        Return:
          shuffled data, label and shuffle indices
    """
    idx = np.arange(len(labels))
    np.random.shuffle(idx)
    return data[idx, ...], labels[idx], idx


### after the adaptations , in this stage i don't think we need to change something here.
def get_learning_rate(batch):
    learning_rate = tf.train.exponential_decay(
        BASE_LEARNING_RATE,  # Base learning rate.
        batch * BATCH_SIZE,  # Current index into the dataset.
        DECAY_STEP,  # Decay step.
        DECAY_RATE,  # Decay rate.
        staircase=True)
    learning_rate = tf.maximum(learning_rate, 0.00001)  # CLIP THE LEARNING RATE!!
    return learning_rate


def get_bn_decay(batch):
    bn_momentum = tf.train.exponential_decay(
        BN_INIT_DECAY,
        batch * BATCH_SIZE,
        BN_DECAY_DECAY_STEP,
        BN_DECAY_DECAY_RATE,
        staircase=True)
    bn_decay = tf.minimum(BN_DECAY_CLIP, 1 - bn_momentum)
    return bn_decay


def train():
    with tf.Graph().as_default():
        with tf.device('/gpu:' + str(GPU_INDEX)):
            # pointclouds_pl size: batch_size x num_point x 9
            # labels_pl size: batch_size x num_point
            pointclouds_pl, labels_pl = placeholder_inputs(BATCH_SIZE, CLOUD_SIZE)
            print(pointclouds_pl.shape, labels_pl.shape)
            is_training_pl = tf.placeholder(tf.bool, shape=())

            # Note the global_step=batch parameter to minimize.
            # That tells the optimizer to helpfully increment the 'batch' parameter for you every time it trains.
            batch = tf.Variable(0)  # constructor of a tensor
            bn_decay = get_bn_decay(batch)  # No hard coding :)
            tf.summary.scalar('bn_decay', bn_decay)

            # why do we do convolution on empty tensor
            # Get model and loss
            pred, end_points = get_model(pointclouds_pl, is_training_pl, bn_decay=bn_decay)
            loss = get_loss(pred, labels_pl, end_points)
            tf.summary.scalar('loss', loss)
            print(pred.shape, loss.shape)

            correct = tf.equal(tf.argmax(pred, 1), tf.to_int64(labels_pl))
            accuracy = tf.reduce_sum(tf.cast(correct, tf.float32)) / float(BATCH_SIZE)
            tf.summary.scalar('accuracy', accuracy)

            # Get training operator
            learning_rate = get_learning_rate(batch)
            tf.summary.scalar('learning_rate', learning_rate)
            if OPTIMIZER == 'momentum':
                optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=MOMENTUM)
            elif OPTIMIZER == 'adam':
                optimizer = tf.train.AdamOptimizer(learning_rate)
            train_op = optimizer.minimize(loss, global_step=batch)

            # Add ops to save and restore all the variables.
            saver = tf.train.Saver()

        # Create a session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        config.log_device_placement = True
        sess = tf.Session(config=config)

        # Add summary writers
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'train'),
                                             sess.graph)
        test_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'test'))

        # Init variables
        init = tf.global_variables_initializer()
        sess.run(init, {is_training_pl: True})

        ops = {'pointclouds_pl': pointclouds_pl,
               'labels_pl': labels_pl,
               'is_training_pl': is_training_pl,
               'pred': pred,
               'loss': loss,
               'train_op': train_op,
               'merged': merged,
               'step': batch}

        for epoch in range(MAX_EPOCH):
            log_string('**** EPOCH %03d ****' % (epoch))
            sys.stdout.flush()

            train_one_epoch(sess, ops, train_writer)
            eval_one_epoch(sess, ops, test_writer)

            # Save the variables to disk.
            if epoch % 2 == 0:
                save_path = saver.save(sess, os.path.join(LOG_DIR, "model.ckpt"))
                log_string("Model saved in file: %s" % save_path)


def train_one_epoch(sess, ops, train_writer):
    """ ops: dict mapping from string to tf ops """
    is_training = True

    log_string('----')
    current_data, current_label, _ = shuffle_data(train_data[:, :, :], train_label)
    file_size = current_data.shape[0]
    num_batches = file_size // BATCH_SIZE
    # num_batches = 10 ###added only for slicing training time. will be deleted later
    total_correct = 0
    total_seen = 0
    loss_sum = 0

    for batch_idx in range(num_batches):
        if (batch_idx % 500 == 0):
            print('Current batch/total batch num: %d/%d' % (batch_idx, num_batches))
        start_idx = batch_idx * BATCH_SIZE
        end_idx = (batch_idx + 1) * BATCH_SIZE

        feed_dict = {ops['pointclouds_pl']: current_data[start_idx:end_idx, :, :],
                     ops['labels_pl']: current_label[start_idx:end_idx],
                     ops['is_training_pl']: is_training, }
        summary, step, _, loss_val, pred_val = sess.run(
            [ops['merged'], ops['step'], ops['train_op'], ops['loss'], ops['pred']],
            feed_dict=feed_dict)
        train_writer.add_summary(summary, step)
        pred_val = np.argmax(pred_val, 1)  # how to
        # print("PREDINCTIONS:", pred_val.shape)
        # print(pred_val)

        # print("CURRENT_LABEL:\n", current_label[start_idx:end_idx])

        correct = np.sum(pred_val == current_label[start_idx:end_idx])
        # print("CORRECT = ", correct)
        total_correct += correct
        total_seen += (BATCH_SIZE)
        loss_sum += loss_val

    log_string('mean loss: %f' % (loss_sum / float(num_batches)))
    log_string('accuracy: %f' % (total_correct / float(total_seen)))


def eval_one_epoch(sess, ops, test_writer):
    """ ops: dict mapping from string to tf ops """
    is_training = False
    total_correct = 0
    total_seen = 0
    loss_sum = 0
    total_seen_class = [0 for _ in range(NUM_CLASSES)]
    total_correct_class = [0 for _ in range(NUM_CLASSES)]

    log_string('----')
    current_data = test_data[:, :, :]
    current_label = np.squeeze(test_label)

    file_size = current_data.shape[0]
    num_batches = file_size // BATCH_SIZE
    # num_batches = 10 ###added only for slicing training time. will be deleted later
    for batch_idx in range(num_batches):
        if (batch_idx % 500 == 0):
            print('Current batch/total batch num: %d/%d' % (batch_idx, num_batches))
        start_idx = batch_idx * BATCH_SIZE
        end_idx = (batch_idx + 1) * BATCH_SIZE

        feed_dict = {ops['pointclouds_pl']: current_data[start_idx:end_idx, :, :],
                     ops['labels_pl']: current_label[start_idx:end_idx],
                     ops['is_training_pl']: is_training}
        summary, step, loss_val, pred_val = sess.run([ops['merged'], ops['step'], ops['loss'], ops['pred']],
                                                     feed_dict=feed_dict)
        test_writer.add_summary(summary, step)
        pred_val = np.argmax(pred_val, 1)
        correct = np.sum(pred_val == current_label[start_idx:end_idx])
        total_correct += correct
        total_seen += (BATCH_SIZE)
        loss_sum += (loss_val * BATCH_SIZE)
        for i in range(start_idx, end_idx):
            l = int(current_label[i])
            total_seen_class[l] += 1
            total_correct_class[l] += (pred_val[i - start_idx] == l)

        

    log_string('eval mean loss: %f' % (loss_sum / float(total_seen)))
    log_string('eval accuracy: %f' % (total_correct / float(total_seen)))
    log_string('eval avg class acc: %f' % (
        np.mean(np.array(total_correct_class) / np.array(total_seen_class, dtype=np.float))))


if __name__ == "__main__":
    train()
    LOG_FOUT.close()
