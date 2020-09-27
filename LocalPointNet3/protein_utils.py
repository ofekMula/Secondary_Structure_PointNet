import sys
import protein_parser
import os
import random
import numpy as np
from Bio.PDB import *
import argparse


"""
this script desinged to supply functions for working with proteins which loaded from 
PDB file format.
"""
# -----------------------------------------------------------------------------
# CONSTANTS
# -----------------------------------------------------------------------------
NUM_POINTS = 512 #
label_dict = {'H': 0,
              'I': 1,
              'G': 2,
              'B': 3,
              'E': 4,
              'S': 5,
              'T': 6,
              '-': 7  # None
              }



parser = argparse.ArgumentParser()
parser.add_argument('--num_proteins', type=int, default=100, help='number of proteins to download [deafult: 100]')
FLAGS, args = parser.parse_known_args()
NUM_PROTEINS = FLAGS.num_proteins
# -----------------------------------------------------------------------------
# PROCESS PROTEINS
# -----------------------------------------------------------------------------

"""downloads the pdb file and extracts the protein structure from it.
We get a protein like "1a09", a chain number like 'A'.
That we download the pdb file & parse it.
returns structure, chain_prot """
def download_and_parse_pdb(pdb_name, num_chain):
    print("parsing_file:", pdb_name, num_chain)
    parser = PDBParser()
    pdbl = PDBList()

    #  Downloading the file
    path = '.'
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


"""create a num_py array of [x,y,z] coordinates
 for all c-alpha atoms in the backbone."""
def list_of_residue_coordinates_and_residue_seq(chain_prot):
    res = protein_parser.process_chain(chain_prot)
    seq = res[0]
    backbone = res[1]
    res = np.array([c[2] for c in backbone])
    return seq, res


"""returns the labels (string) 
of secondary structure for each residue."""
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
            if (protein_parser.is_residue(residue)):
                if residue.get_id()[1] == residue_number:
                    secondary_struct += secondary_structure_of_residue
                    if secondary_structure_of_residue in label_dict:
                        label_array = np.append(label_array, label_dict[secondary_structure_of_residue])  # label
                    # every position in the array corresponds to residue coordinates in the same position
                    else:
                        label_array = np.append(label_array, label_dict['-'])
                    break

    return secondary_struct, label_array

"""gets the list of coordinates of the c-alpha atoms,
normalize them to values between -1 to 1.
returns the normalized coordinates"""
def normalize_coordinates(list_residues_coord):
  #make every coordinate between -1 and 1
  coord_abs = np.absolute(list_residues_coord)
  xyz_max = np.amax(coord_abs, axis=0)[0:3]
  list_residues_coord[:, 0:3] /= xyz_max
  return list_residues_coord

"""gets the protein name,array of data (coordinates or labels) and file name
and write the data to a numpy file format with the given name.
protein_name for example: 11as_A"""
def save_to_numpy_file(protein_name, array_to_save, string):
    out_file_name = './' + protein_name + string  # protein name : 11as_A.numpy
    np.save(out_file_name, array_to_save)
    return out_file_name


""" saves all the given pdbs into a numpy file array which written to the
Protein_Info folder.
list of pdbs = 'pdbs_list.txt'
"""
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
        protein_file_name = pdb_name + '_' + num_chain
        structure, chain_prot = download_and_parse_pdb(pdb_name, num_chain)
        if (isinstance(structure, int)):
          print("ERROR: pdb not found")
          continue
        list_of_residues, list_residue_coord = list_of_residue_coordinates_and_residue_seq(chain_prot)
        full = normalize_coordinates(list_residue_coord)
        _, list_of_labels = list_of_residue_labels(pdb_name, structure, num_chain, chain_prot)
        if(full.shape[0] != list_of_labels.shape[0]):
          print("different lengths!")
          os.remove(pdb_name + ".pdb")
          continue
        os.mkdir("./Proteins_Info/" + pdb_name + "_" + num_chain)
        out_file_name = save_to_numpy_file(protein_file_name, full, '_data') + '.npy'
        os.rename(out_file_name,
                  "./Proteins_Info/" + pdb_name + "_" + num_chain + "/"
                  + pdb_name + '_' + num_chain + '_data.npy')
        out_file_name = save_to_numpy_file(protein_file_name, list_of_labels, '_label') + '.npy'
        os.rename(out_file_name,
                  "./Proteins_Info/" + pdb_name + "_" + num_chain + "/"
                  + pdb_name + '_' + num_chain + '_label.npy')

        os.remove(pdb_name + ".pdb")
        proteins_in_Proteins_Info.add(folder_name)
        number_of_structures += 1
        if (number_of_structures % 10 == 0):
          print("\t\t\tNUMBER OF STRUCTURES = ", number_of_structures)



if __name__ == "__main__":
    save_numpy_files_from_proteins("list_pdbs.txt", NUM_PROTEINS)

