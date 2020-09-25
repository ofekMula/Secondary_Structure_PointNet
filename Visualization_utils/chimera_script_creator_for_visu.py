import os
from Bio.PDB import Selection
import numpy as np
from protein_utils import *
import argparse

"""
this file is for creating chimera script for coloring a protein by its secondary structure
please be ware of the command line arguments 
requirements :protein_util.py file in the same directory
you can find it in any of the PointNet folders
"""

parser = argparse.ArgumentParser()
parser.add_argument('--pdb_name', type=str, default="5w1o", help='pdb name for visu')
parser.add_argument('--num_chain', type=str, default="A", help='num_chain for visu')
parser.add_argument('--npy_filename', type=str, default="",
                    help='filename of numpy array of secondary structure labels')
FLAGS = parser.parse_args()

PDB_NAME = FLAGS.pdb_name
NUM_CHAIN = FLAGS.num_chain
NPY_FILENAME = FLAGS.npy_filename

residue_dictionary = {'CYS': 'C', 'ASP': 'D', 'SER': 'S', 'GLN': 'Q', 'LYS': 'K',
                      'ILE': 'I', 'PRO': 'P', 'THR': 'T', 'PHE': 'F', 'ASN': 'N',
                      'GLY': 'G', 'HIS': 'H', 'LEU': 'L', 'ARG': 'R', 'TRP': 'W',
                      'ALA': 'A', 'VAL': 'V', 'GLU': 'E', 'TYR': 'Y', 'MET': 'M', 'MSE': 'M'}


def is_residue(residue):
    try:
        return (residue.get_id()[0] == ' ') & (residue.resname in residue_dictionary.keys())
    except:
        return False


def get_PDB_indices(chain_obj):
    list_indices = []
    for residue in Selection.unfold_entities(chain_obj, 'R'):
        if is_residue(residue):
            list_indices.append((residue.get_full_id()[1], residue.get_full_id()[2], residue.get_id()[1]))
    list_indices = np.array(list_indices)
    return list_indices


def make_script_for_visu_predictions(chain, labels, output_name=PDB_NAME + "_" + NUM_CHAIN + "_" + 'chimera_script.py'):
    colors = [
        'medium blue',
        'yellow',
        'red',
        'tan',
        'blue',
        'orange',
        'orange red',
        'cornflower blue']

    trans_label = {0: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 2, 6: 2, 7: 2}
    residue_color = [colors[trans_label[int(label)]] for label in labels]
    # residue_color = [colors[int(label)] for label in labels]
    list_indices = get_PDB_indices(chain)
    print(list_indices.shape)

    print(len(residue_color))
    pdb = chain.get_full_id()[0]
    L = len(labels)

    with open(output_name, 'w') as f:
        f.write('import chimera\n')
        f.write('from chimera import runCommand\n')
        list_commands = []
        list_commands.append('open %s' % pdb)
        list_commands.append('color dark gray #')
        for l in range(L):
            list_commands.append('color %s #%s.%s:%s.%s' % (
                residue_color[l],
                0,
                list_indices[l, 0],
                list_indices[l, 2],
                list_indices[l, 1]))
        for command in list_commands:
            f.write("runCommand('%s')\n" % command)
    return output_name


if __name__ == "__main__":

    pdb_name = "5w1o"
    pdb_name = "3h3m"
    num_chain = 'A'
    # num_chain='L'
    pdb_filename = pdb_name + ".pdb"
    structure, chain_obj = download_and_parse_pdb(PDB_NAME, NUM_CHAIN)
    if NPY_FILENAME == "":
        _, labels = list_of_residue_labels(PDB_NAME, structure, NUM_CHAIN, chain_obj)
    else:
        lables = np.load(NPY_FILENAME)
    make_script_for_visu_predictions(chain_obj, labels)