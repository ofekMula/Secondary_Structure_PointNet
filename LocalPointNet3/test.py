import argparse
import protein_utils
import tensorflow as tf
import os
import sys
from pointnet_cls import *
import local_point_clouds

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

NUM_CLASSES = 3
CLOUD_SIZE=8

def log_string(out_str):
    LOG_FOUT.write(out_str + '\n')
    LOG_FOUT.flush()
    print(out_str)


def evaluate():
    is_training = False

    with tf.device('/gpu:' + str(GPU_INDEX)):
        pointclouds_pl, labels_pl = placeholder_inputs(BATCH_SIZE, CLOUD_SIZE)
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

    structure, chain_prot = protein_utils.download_and_parse_pdb(pdb_name, num_chain)
    if isinstance(structure, int):
        print("ERROR: pdb not found")
        sys.exit()
    print(pdb_name)
    list_of_residues, list_residue_coord = protein_utils.list_of_residue_coordinates_and_residue_seq(chain_prot)
    full = protein_utils.normalize_coordinates(list_residue_coord)
    ##We try (-1, -1, ..., -1) lables (garbage)
    list_of_labels = np.array([1 for i in full])
    #list_coord, list_labels = protein_utils.fill_arrays_to_num_points(full, list_of_labels)
    list_coord, list_labels = full, list_of_labels
    print(list_coord.shape, list_labels.shape)
    
    #Centerelizing the proteing to (0,0,0)
    xyz_mean = np.mean(list_coord, axis=0)[0:3]
    list_coord[:, 0:3] -= xyz_mean

    #Instead of n X 3, we want to have n*32*3
    local_data = []
    local_label = []
    local_cloud, _ = local_point_clouds.build_local_point_cloud(list_coord, CLOUD_SIZE)
    num_points_in_protein = list_coord.shape[0]
    #neigbors is of shape num_of_points X 32.
    for j in range(num_points_in_protein):
        local_data.append(local_cloud[j])
        local_label.append(list_labels[j])
    local_data = np.array(local_data)
    local_label = np.array(local_label)
    print(local_data.shape)
    print(local_label.shape)

    #new_diminsional_coord_list = np.array([list_coord])
    #new_diminsional_label_list = np.array([list_labels])
    #print(new_diminsional_coord_list.shape, new_diminsional_label_list.shape)
    
    
    os.remove(pdb_name + ".pdb")
    pred_label = []
    number_of_points_in_protein = local_data.shape[0]
    print(number_of_points_in_protein, "points")
    for j in range(number_of_points_in_protein):
          feed_dict = {ops['pointclouds_pl']: [local_data[j]],
          ops['labels_pl']: [local_label[j]],
          ops['is_training_pl']: is_training}
          loss_val, pred_val = sess.run([ops['loss'], ops['pred_softmax']],
                                        feed_dict=feed_dict)
          pred_residue_label = np.argmax(pred_val, 1)  # BxN
          #print("PRED_LABEL:\n ", pred_label)
          pred_label.append(pred_residue_label)
      
    pred_label = np.array([pred_label[i][0] for i in range(len(pred_label))])
    print(len(pred_label))
    print(pred_label)



    fout_out.write(out_data_label_filename + '\n')
    fout_out.close()


if __name__ == '__main__':
    with tf.Graph().as_default():
        evaluate()
    LOG_FOUT.close()
