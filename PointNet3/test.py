import argparse
from protein_utils import *
import tensorflow as tf
import os
import sys
from pointnet_seg import *

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--batch_size', type=int, default=1, help='Batch Size during training [default: 1]')
parser.add_argument('--model_path', required=True, help='model checkpoint file path')
parser.add_argument('--dump_dir', required=True, help='dump folder path')
parser.add_argument('--output_file', required=True, help='TXT filename, filelist, each line is an output for a room')
parser.add_argument('--input_pdb_chain', required=True, help='PDB-id_chain filename')

FLAGS = parser.parse_args()

BATCH_SIZE = FLAGS.batch_size
MODEL_PATH = FLAGS.model_path
GPU_INDEX = FLAGS.gpu
DUMP_DIR = FLAGS.dump_dir
INPUT = FLAGS.input_pdb_chain
FOUT = FLAGS.output_file

if not os.path.exists(DUMP_DIR): os.mkdir(DUMP_DIR)
LOG_FOUT = open(os.path.join(DUMP_DIR, 'log_evaluate.txt'), 'w')
LOG_FOUT.write(str(FLAGS) + '\n')

NUM_POINT = 512
NUM_CLASSES = 3


def log_string(out_str):
    LOG_FOUT.write(out_str + '\n')
    LOG_FOUT.flush()
    print(out_str)


def evaluate():
    is_training = False

    with tf.device('/gpu:' + str(GPU_INDEX)):
        pointclouds_pl, labels_pl = placeholder_inputs(BATCH_SIZE, NUM_POINT)
        is_training_pl = tf.placeholder(tf.bool, shape=())
        print(pointclouds_pl.shape, labels_pl.shape)

        # simple model
        pred, end_points = get_model(pointclouds_pl, is_training_pl)
        loss = get_loss(pred, labels_pl, end_points)
        pred_softmax = tf.nn.softmax(pred)

        # Add ops to save and restore all the variables.

        saver = tf.train.Saver()

    # Create a session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    config.log_device_placement = True
    sess = tf.Session(config=config)

    # Restore variables from disk.
    saver.restore(sess, MODEL_PATH)
    print("Model restored.")

    ops = {'pointclouds_pl': pointclouds_pl,
           'labels_pl': labels_pl,
           'is_training_pl': is_training_pl,
           'pred': pred,
           'pred_softmax': pred_softmax,
           'loss': loss}

    fout_out = open(FOUT, 'w')

    out_data_label_filename = INPUT + '_output_pred.txt'
    out_data_label_filename = os.path.join(DUMP_DIR, out_data_label_filename)
    print("dest: ", out_data_label_filename)

    fout_data_label = open(out_data_label_filename, 'w')

    ## parsing pdb file and loading numpy arrays
    flag = 0
    num_chain = 0
    pdb_name = ""
    for char in INPUT:
        if char != '_':
            if flag == 1:
                num_chain = [c for c in char][0]
                break
            else:
                pdb_name += char
        else:
            flag = 1
            continue

    structure, chain_prot = download_and_parse_pdb(pdb_name, num_chain)
    if isinstance(structure, int):
        print("ERROR: pdb not found")
        sys.exit()
    print(pdb_name)
    list_of_residues, list_residue_coord = list_of_residue_coordinates_and_residue_seq(chain_prot)
    full = normalize_coordinates(list_residue_coord)
    ##We try (-1, -1, ..., -1) lables (garbage)
    list_of_labels = np.array([1 for i in full])
    list_coord, list_labels = fill_arrays_to_num_points(full, list_of_labels)
    print(list_coord.shape, list_labels.shape)

    #Centerelizing the proteing to (0,0,0)
    xyz_mean = np.mean(list_coord, axis=0)[0:3]
    list_coord[:, 0:3] -= xyz_mean
    
    
    new_diminsional_coord_list = np.array([list_coord])
    new_diminsional_label_list = np.array([list_labels])
    print(new_diminsional_coord_list.shape, new_diminsional_label_list.shape)
    os.remove(pdb_name + ".pdb")

    feed_dict = {ops['pointclouds_pl']: new_diminsional_coord_list,
                 ops['labels_pl']: new_diminsional_label_list,
                 ops['is_training_pl']: is_training}
    loss_val, pred_val = sess.run([ops['loss'], ops['pred_softmax']],
                                  feed_dict=feed_dict)
    pred_label = np.argmax(pred_val, 2)  # BxN
    print("PRED_LABEL:\n ", pred_label)

    fout_out.write(out_data_label_filename + '\n')
    fout_out.close()


if __name__ == '__main__':
    with tf.Graph().as_default():
        evaluate()
    LOG_FOUT.close()
