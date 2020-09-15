import sys
import sample_code
import os
import random
import argparse
# from Bio.PDB.DSSP import DSSP
import numpy as np
from Bio.PDB import *

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)

# -----------------------------------------------------------------------------
# CONSTANTS
# -----------------------------------------------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument('--num_proteins', type=int, default=1000, help='Database size [default:1000]')
FLAGS = parser.parse_args()
NUM_PROTEINS = FLAGS.num_proteins


NUM_POINTS = 512

label_dict = {'H': 0,
              'I': 1,
              'G': 2,
              'B': 3,
              'E': 4,
              'S': 5,
              'T': 6,
              '-': 7  # None
              }

residue_to_attributes = {'C': np.array([28.0, 7.4, 26.3, 40.3, 7.4, 0]),
                         'D': np.array([31.3, 100.0, 0.0, 17.5, 45.0, 0]),
                         'S': np.array([18.1, 32.1, 34.8, 1.9, 40.5, 0]),
                         'Q': np.array([51.3, 45.7, 34.4, 0.0, 43.6, 0]),
                         'K': np.array([68.0, 64.2, 86.9, 43.5, 54.3, 0]),
                         'I': np.array([63.6, 0.0, 39.2, 83.6, 7.5, 0]),
                         'P': np.array([41.0, 21.0, 40.2, 73.5, 66.2, 0]),
                         'T': np.array([34.0, 21.0, 45.7, 1.9, 35.3, 0]),
                         'F': np.array([77.2, 1.2, 38.6, 76.1, 5.5, 1]),
                         'N': np.array([35.4, 63.0, 31.3, 2.4, 46.1, 0]),
                         'G': np.array([0.0, 37.0, 38.5, 2.7, 54.0, 0]),
                         'H': np.array([49.2, 43.2, 59.2, 23.1, 28.1, 0]),
                         'L': np.array([63.6, 0.0, 38.6, 57.6, 10.1, 0]),
                         'R': np.array([70.8, 51.9, 100.0, 22.6, 50.1, 0]),
                         'W': np.array([100.0, 4.9, 37.7, 100.0, 13.8, 1]),
                         'A': np.array([15.9, 25.9, 39.2, 23.1, 37.4, 0]),
                         'V': np.array([47.7, 8.6, 38.5, 49.6, 19.6, 0]),
                         'E': np.array([47.2, 93.8, 3.2, 17.8, 48.6, 0]),
                         'Y': np.array([78.5, 9.9, 34.4, 70.8, 30.1, 1]), 'M': np.array([62.8, 4.9, 35.7, 44.3, 3.9, 0])
                         }


# -----------------------------------------------------------------------------
# PROCESS PROTEINS
# -----------------------------------------------------------------------------
##  We get a protein like "1a09", a chain number like 'A'.
##  That we download the pdb file & parse it.


def download_and_parse_pdb(pdb_name, num_chain):
    print("parsing_file:", pdb_name, num_chain)
    parser = PDBParser()
    pdbl = PDBList()

    #  Downloading the file
    path = "."
    file_name = pdbl.retrieve_pdb_file(pdb_code=pdb_name, file_format="pdb", pdir=path)
    new_name = path + '/' + pdb_name + '.pdb'
    if (not os.path.isfile(file_name)):
      return -1, -1
    os.rename(file_name, new_name)
    file_data_name = pdb_name + '.pdb'

    #  Parsing the file
    structure = parser.get_structure(pdb_name, file_data_name)
    chain_prot = structure[0][num_chain]

    return structure, chain_prot


#  create a num_py array of [x,y,z] coordinates for all atoms.
def list_of_residue_coordinates_and_residue_seq(chain_prot):
    res = sample_code.process_chain(chain_prot)
    seq = res[0]
    backbone = res[1]
    res = np.array([c[2] for c in backbone])
    return seq, res


# create a string of secondary structure for each residue.
def list_of_residue_labels(pdb_name, structure, num_chain, chain_prot):
    file_data_name = pdb_name + '.pdb'
    model = structure[0]
    dssp = DSSP(model, file_data_name, dssp='mkdssp')
    secondary_struct = ""
    label_array = np.array([])

    for i in range(len(list(dssp.keys()))):
        if dssp.keys()[i][0] != num_chain:
            continue
        residue_number = dssp.keys()[i][1][1]
        # find relevant key in dictionary
        a_key = list(dssp.keys())[i]
        secondary_structure_of_residue = dssp[a_key][2]
        # find how many atoms in this residue
        for residue in Selection.unfold_entities(chain_prot, 'R'):
            if (sample_code.is_residue(residue)):
                if residue.get_id()[1] == residue_number:
                    secondary_struct += secondary_structure_of_residue
                    if secondary_structure_of_residue in label_dict:
                        label_array = np.append(label_array, label_dict[secondary_structure_of_residue])  # label
                    # every position in the array corresponds to residue coordinates in the same position
                    else:
                        label_array = np.append(label_array, label_dict['-'])
                    break

    return secondary_struct, label_array


def create_cord_label_array(coordinates, labels):
    if len(coordinates) != len(labels):
        print("ERROR - different number of coordinates and colors")
    else:
        arr = np.array([])
        for i in range(0, len(coordinates)):
            arr = np.array(arr, [coordinates[i], labels[i]])  # add current coordinates with there color
        return arr


## receives list_residues
def normalize_coordinates(list_residues_coord):
  #make every coordinate between -1 and 1
  coord_abs = np.absolute(list_residues_coord)
  xyz_max = np.amax(coord_abs, axis=0)[0:3]
  list_residues_coord[:, 0:3] /= xyz_max
  return list_residues_coord


## receives the residues sequence and noralizes
## zi = (xi - min(x))/(max(x)-min(x))
def normalize_attributes(list_residues_sequence):
    attribute_list = np.array([residue_to_attributes[c] for c in list_residues_sequence])
    attribute_max = np.amax(attribute_list, axis=0)[0:6]
    attribute_min = np.amin(attribute_list, axis=0)[0:6]

    attribute_list[:, 0:6] -= attribute_min
    attribute_list[:, 0:6] /= (attribute_max - attribute_min)
    return attribute_list


def normalize_full_attributes(list_residues_sequence, list_residues_coord):
    attribute_list = normalize_attributes(list_residues_sequence)
    list_residues_coord = normalize_coordinates(list_residues_coord)
    if len(attribute_list) != len(list_residues_coord):
        print("ERROR: Protein_Utils, normalize_full_attributes: attr len != len of coordinates")
        return -1
    else:
        len_arr = len(attribute_list)
        full_attr_array = np.array(
            [np.concatenate((list_residues_coord[i], attribute_list[i])) for i in range(len_arr)])
        return full_attr_array


# protein_path for example: Area_1/11as_A/Protein
def save_to_numpy_file(protein_path, array_to_save, string):
    out_file_name = BASE_DIR + '/' + str.split(protein_path, '/')[1] + string  # protein name : 11as_A.numpy
    np.save(out_file_name, array_to_save)
    return out_file_name


# list of pdbs = "pdbs_list.txt"
def save_numpy_files_from_proteins(list_of_pdbs, number_of_proteins):
    proteins_in_Proteins_Info = set()
    if os.path.isdir("Proteins_Info"):
        for subdir, dirs, files in os.walk("Proteins_Info"):
          for protein_dir in dirs:
            proteins_in_Proteins_Info.add(protein_dir)
    else:
        os.mkdir("Proteins_Info")

    f_read = open(list_of_pdbs, "r")
    number_of_structures = 0
    for line in f_read:
        flag = 0
        num_chain = 0
        pdb_name = ""
        if number_of_structures == number_of_proteins:
            break
        for char in line:
            if char != "_":
                if flag == 1:
                    num_chain = [c for c in char][0]
                    break
                else:
                    pdb_name += char
            else:
                flag = 1
                continue
        

        #check if we already have this protein
        folder_name = pdb_name + "_" + num_chain
        if folder_name in proteins_in_Proteins_Info:
          continue
        else:
          print("NAME = ", pdb_name, "CHAIN = ", num_chain)


        ##Now creating the file
        protein_file_name = 'Area_1/' + pdb_name + '_' + num_chain + '/Protein'
        structure, chain_prot = download_and_parse_pdb(pdb_name, num_chain)
        if (isinstance(structure, int)):
          print("ERROR: pdb not found")
          continue
        list_of_residues, list_residue_coord = list_of_residue_coordinates_and_residue_seq(chain_prot)
        full = normalize_coordinates(list_residue_coord)
        _, list_of_labels = list_of_residue_labels(pdb_name, structure, num_chain, chain_prot)
        list_coord, list_labels = fill_arrays_to_num_points(full, list_of_labels)
        if (list_coord.shape[0] == 0 or list_labels.shape[0] == 0):
          os.remove(pdb_name + ".pdb")
          continue
        os.mkdir("./Proteins_Info/" + pdb_name + "_" + num_chain)
        out_file_name = save_to_numpy_file(protein_file_name, list_coord, '_data') + '.npy'
        os.rename(out_file_name,
                  "./Proteins_Info/" + pdb_name + "_" + num_chain + "/"
                  + pdb_name + '_' + num_chain + '_data.npy')
        out_file_name = save_to_numpy_file(protein_file_name, list_labels, '_label') + '.npy'
        os.rename(out_file_name,
                  "./Proteins_Info/" + pdb_name + "_" + num_chain + "/"
                  + pdb_name + '_' + num_chain + '_label.npy')

        os.remove(pdb_name + ".pdb")
        proteins_in_Proteins_Info.add(folder_name)
        number_of_structures += 1
        if (number_of_structures % 10 == 0):
          print("\t\t\tNUMBER OF STRUCTURES = ", number_of_structures)


# Gets list_coord(327X9) and list_labels(327) numpy.
# Duplicates random indexes in the arrays correspondly.
# If arrays' length is bigger than NUM_POINTS, we remove correspond values.
def fill_arrays_to_num_points(list_coord, list_labels):
    length = list_coord.shape[0]
    if (length != list_labels.shape[0]):
        print("ERROR: fill_arrays() with different lengths")
        return np.array([]), np.array([])
    # CASE1: fill in points
    if (length < NUM_POINTS):
        to_fill = NUM_POINTS - length
        for i in range(to_fill):
            index = random.randint(0, length - 1)
            list_coord = np.append(list_coord, [list_coord[index]], axis=0)
            list_labels = np.append(list_labels, [list_labels[index]], axis=0)
    # CASE2: remove somve points
    if (length > NUM_POINTS):
        list_coord = list_coord[0:NUM_POINTS]
        list_labels = list_labels[0:NUM_POINTS]

    return list_coord, list_labels


if __name__ == "__main__":
    save_numpy_files_from_proteins("list_pdbs.txt", NUM_PROTEINS)

