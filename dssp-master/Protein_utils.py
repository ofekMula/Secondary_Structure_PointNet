import numpy as np
import glob
import os
import sys
import sample_code
import os
# from Bio.PDB.DSSP import DSSP
import Bio.PDB
import numpy as np
from Bio.PDB import *

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)

# -----------------------------------------------------------------------------
# CONSTANTS
# -----------------------------------------------------------------------------

label_dict = { 'H': 0,
                'I': 1,
                'G': 2,
                'B': 3,
                'E': 4,
                'S': 5,
                'T': 6,
                '-': 7  # None
                }

residue_to_attributes = {'C': np.array([28.0, 7.4, 26.3, 40.3, 7.4, 0]), 'D': np.array([31.3, 100.0, 0.0, 17.5, 45.0, 0]),
                         'S': np.array([18.1, 32.1, 34.8, 1.9, 40.5, 0]), 'Q': np.array([51.3, 45.7, 34.4, 0.0, 43.6, 0]),
                         'K': np.array([68.0, 64.2, 86.9, 43.5, 54.3, 0]), 'I': np.array([63.6, 0.0, 39.2, 83.6, 7.5, 0]),
                         'P': np.array([41.0, 21.0, 40.2, 73.5, 66.2, 0]), 'T': np.array([34.0, 21.0, 45.7, 1.9, 35.3, 0]),
                         'F': np.array([77.2, 1.2, 38.6, 76.1, 5.5, 1]), 'N': np.array([35.4, 63.0, 31.3, 2.4, 46.1, 0]),
                         'G': np.array([0.0, 37.0, 38.5, 2.7, 54.0, 0]), 'H': np.array([49.2, 43.2, 59.2, 23.1, 28.1, 0]),
                         'L': np.array([63.6, 0.0, 38.6, 57.6, 10.1, 0]), 'R': np.array([70.8, 51.9, 100.0, 22.6, 50.1, 0]),
                         'W': np.array([100.0, 4.9, 37.7, 100.0, 13.8, 1]), 'A': np.array([15.9, 25.9, 39.2, 23.1, 37.4, 0]),
                         'V': np.array([47.7, 8.6, 38.5, 49.6, 19.6, 0]), 'E': np.array([47.2, 93.8, 3.2, 17.8, 48.6, 0]),
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
    path = "/specific/netapp5_2/iscb/wolfson/ofekm/v_env/dssp-master"
    file_name = pdbl.retrieve_pdb_file(pdb_code=pdb_name, file_format="pdb", pdir=path)
    new_name = path + '/' + pdb_name + '.pdb'
    os.rename(file_name, new_name)
    file_data_name = pdb_name + '.pdb'

    #  Parsing the file
    structure = parser.get_structure(pdb_name, file_data_name)
    chain_prot = structure[0][num_chain]

    return structure, chain_prot


#  create a num_py array of [x,y,z] coordinates for all atoms.
# def list_of_atom_coordinates(chain_prot):
#     res = sample_code.process_chain(chain_prot)
#     all_coordinates = res[2]
#     list_of_atoms = [np.zeros(3)]
#     for i in range(len(all_coordinates)):
#         for j in range(len(all_coordinates[i])):
#             list_of_atoms = np.append(list_of_atoms, [all_coordinates[i][j]], axis=0)
#     return np.delete(list_of_atoms, 0, axis=0)


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
        # print("res_num:", residue_number)
        # find relecant key in dictionary
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

# create a string of secondary structure for each atom.
# def list_of_atom_colors(pdb_name, structure, num_chain, chain_prot):
#     file_data_name = pdb_name + '.pdb'
#     model = structure[0]
#     dssp = DSSP(model, file_data_name, dssp='mkdssp')
#     secondary_struct = ""
#     colors_array = np.array([])
#
#     for i in range(len(list(dssp.keys()))):
#         if dssp.keys()[i][0] != num_chain:
#             continue
#         residue_number = dssp.keys()[i][1][1]
#         # print("res_num:", residue_number)
#         # find relecant key in dictionary
#         a_key = list(dssp.keys())[i]
#         secondary_structure_of_residue = dssp[a_key][2]
#         # find how many atoms in this residue
#         for residue in Selection.unfold_entities(chain_prot, 'R'):
#             if (sample_code.is_residue(residue)):
#                 if residue.get_id()[1] == residue_number:
#                     num_of_atoms_in_residue = len([atom for atom in residue if sample_code.is_heavy_atom(atom)])
#                     secondary_struct += num_of_atoms_in_residue * secondary_structure_of_residue
#                     if secondary_structure_of_residue in colors_dict:
#                         colors_array = np.array(colors_array, colors_dict[secondary_structure_of_residue])  # it is an array
#                     # of arrays of size 3 to represent colors
#                     # every position in the array corresponds to atom coordinates in the same position
#                     else:
#                         colors_array = np.array(colors_array, colors_dict['-'])
#                     break
#         # add to output
#
#     return secondary_struct, colors_array


# protein_path for example: Area_1/11as_A/Protein
def save_to_numpy_file(protein_path, point_cord_color_array):
    out_file_name = BASE_DIR + str.split(protein_path, '/')[1] + '.numpy'  # protein name : 11as_A.numpy
    np.save(out_file_name, point_cord_color_array)


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
    xyz_max = np.amax(list_residues_coord, axis=0)[0:3]
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
        full_attr_array = np.array([np.concatenate((list_residues_coord[i], attribute_list[i])) for i in range(len_arr)])
        return full_attr_array





if __name__ == "__main__":
    #protein_file_name = 'Area_1/11as_A/Protein'
    structure, chain_prot = download_and_parse_pdb("11as", 'A')
    list_of_residues, list_residue_coord = list_of_residue_coordinates_and_residue_seq(chain_prot)
    print("SEQ LEN = ", len(list_of_residues))
    print(list_of_residues)
    print("COORD LEN = ", len(list_residue_coord))
    print(list_residue_coord)
    norm_coord = normalize_coordinates(list_residue_coord)
    print ("NORM COORD LEN =", len(norm_coord))
    print(norm_coord)
    full = normalize_full_attributes(list_of_residues, list_residue_coord)
    print("FULL LEN = ", len(full))
    print(full)
    print(full.shape)
    sec_struct, list_of_colors = list_of_residue_labels("11as", structure, 'A', chain_prot)
    print("COLORS LEN = ", len(list_of_colors))
    print(sec_struct)
    print(list_of_colors)

    # list_of_atoms = list_of_atom_coordinates(chain_prot)
    # print(list_of_atoms, "\nlen = ", len(list_of_atoms))
    # list_of_colors, colors_arr = list_of_atom_colors("11as", structure, 'A', chain_prot)
    # print(list_of_colors, "\nlen = ", len(list_of_colors))
    #
    # cord_color_arr = create_cord_color_array(list_of_atoms, colors_arr)
    # save_to_numpy_file(protein_file_name, cord_color_arr)
