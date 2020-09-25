import os
from Bio.PDB.DSSP import DSSP
import Bio.PDB
import numpy as np
from Bio.PDB import *

list_atoms_types = ['C', 'O', 'N', 'S']  # H
VanDerWaalsRadii = np.array([1.70, 1.52, 1.55, 1.80])  # 1.20

atom_mass = np.array(
    [
        12,  # C
        16,  # O
        14,  # N
        32  # S
    ]
)

atom_type_to_index = dict([(list_atoms_types[i], i)
                           for i in range(len(list_atoms_types))])

list_atoms = ['C', 'CA', 'CB', 'CD', 'CD1', 'CD2', 'CE', 'CE1', 'CE2', 'CE3',
              'CG', 'CG1', 'CG2', 'CH2', 'CZ', 'CZ2', 'CZ3', 'N', 'ND1', 'ND2',
              'NE', 'NE1', 'NE2', 'NH1', 'NH2', 'NZ', 'O', 'OD1', 'OD2', 'OE1',
              'OE2', 'OG', 'OG1', 'OH', 'OXT', 'SD', 'SE', 'SG']

atom_to_index = dict([(list_atoms[i], i) for i in range(len(list_atoms))])

atom_to_index['OT1'] = atom_to_index['O']
atom_to_index['OT2'] = atom_to_index['O']

residue_dictionary = {'CYS': 'C', 'ASP': 'D', 'SER': 'S', 'GLN': 'Q', 'LYS': 'K',
                      'ILE': 'I', 'PRO': 'P', 'THR': 'T', 'PHE': 'F', 'ASN': 'N',
                      'GLY': 'G', 'HIS': 'H', 'LEU': 'L', 'ARG': 'R', 'TRP': 'W',
                      'ALA': 'A', 'VAL': 'V', 'GLU': 'E', 'TYR': 'Y', 'MET': 'M', 'MSE': 'M'}


def is_residue(residue):
    try:
        return (residue.get_id()[0] == ' ') & (residue.resname in residue_dictionary.keys())
    except:
        return False


def is_heavy_atom(atom):
    # Second condition for Rosetta-generated files.
    try:
        return (atom.get_id()[0] in atom_type_to_index.keys()) & (atom.get_id() != 'CEN')
    except:
        return False


def process_chain(chain):
    sequence = ''
    backbone_coordinates = []
    all_coordinates = []
    all_atoms = []
    all_atom_types = []
    for residue in Selection.unfold_entities(chain, 'R'):
        if is_residue(residue):
            sequence += residue_dictionary[residue.resname]
            residue_atom_coordinates = np.array(
                [atom.get_coord() for atom in residue if is_heavy_atom(atom)])
            residue_atoms = [atom_to_index[atom.get_id()]
                             for atom in residue if is_heavy_atom(atom)]
            residue_atom_type = [atom_type_to_index[atom.get_id()[0]]
                                 for atom in residue if is_heavy_atom(atom)]
            residue_backbone_coordinates = []
            for atom in ['N', 'C', 'CA', 'O', 'CB']:
                try:
                    residue_backbone_coordinates.append(
                        residue_atom_coordinates[residue_atoms.index(atom_to_index[atom])])
                except:
                    residue_backbone_coordinates.append(
                        np.ones(3, dtype=np.float32) * np.nan)
            backbone_coordinates.append(residue_backbone_coordinates)
            all_coordinates.append(residue_atom_coordinates)
            all_atoms.append(residue_atoms)
            all_atom_types.append(residue_atom_type)
    backbone_coordinates = np.array(backbone_coordinates)
    return sequence, backbone_coordinates, all_coordinates, all_atoms, all_atom_types


if __name__ == "__main__":
    num_of_structures = 0
    parser = PDBParser()
    pdbl = PDBList()
    output = "pdb_data.txt"
    f_read = open("list_pdbs.txt", "r")  # before i deleted the 1st row in file - need to bring it back
    f_write = open("pdb_data.txt", "w")

    for line in f_read:
        print("line:", line)
        flag = 0
        num_chain = 0
        pdb_name = ""
        if num_of_structures == 5:
            break

        #########################################
        #Downloading PDB files
        #########################################
        for char in line:
            #print(char)
            if char != "_":
                if flag == 1:
                    #num_chain = ord(char)-ord('A')  # get num of chain
                    num_chain = [c for c in char][0]
                    break  # finished pdb name and chain
                else:
                    pdb_name += char  # concat the char to name
                    #print("pdb", pdb_name)
            else:
                flag = 1  # end of pdb name
                continue
        # got the pdb name:
        # before the next stage need to download the structure
        print("NAME = ", pdb_name, "CHAIN = ", num_chain)
        # CHANGE - path
        # path = "C:\\Users\\Ricky Benkovich\\biopythontry1"
        path = "/specific/netapp5_2/iscb/wolfson/yarinluhmany/v_env/dssp-master"
        file_name = pdbl.retrieve_pdb_file(pdb_code=pdb_name, file_format="pdb", pdir=path)
        new_name = path + '/' + pdb_name + '.pdb'
        os.rename(file_name, new_name)
        file_data_name = pdb_name+'.pdb'
        #########################################
        #Parsing PDB files
        #########################################
        structure = parser.get_structure(pdb_name, file_data_name)
        #chain_prot = Selection.unfold_entities(structure, 'C')[num_chain]
        chain_prot = structure[0][num_chain]
        sequence, backbone_coordinates, all_coordinates, all_atoms, all_atom_types = process_chain(chain_prot)
        f_write.write("\n"+pdb_name+"\n")
        f_write.write(sequence)

##        #########################################
##        #Printing backbone coordinates
##        #########################################
##        #f_write.write("\n[")
##        #print("[")
##        #for i in range(len(backbone_coordinates)):
##            #print("[")
##        #    f_write.write("[")
##        #    for j in range(len(backbone_coordinates[i])):
##                f_write.write(str(backbone_coordinates[i][j]))
##                #print(backbone_coordinates[i][j])
##            #print("]")
##            f_write.write("]")
##        f_write.write("]")
##        print("]")
        
        #########################################
        #Printing atomic coordinates
        #########################################
        f_write.write("\n[")
        #print("[")
        for i in range(len(all_coordinates)):
            #print("[")
            f_write.write("[")
            for j in range(len(all_coordinates[i])):
                f_write.write(str(all_coordinates[i][j])+" ")
                #print(all_coordinates[i][j])
            #print("]")
            f_write.write("]")
        f_write.write("]")
        #print("]")


        #########################################
        #Printing atomic IDs
        #########################################
        f_write.write("\n[")
        #print("[")
        for i in range(len(all_atoms)):
            #print("[")
            f_write.write("[")
            for j in range(len(all_atoms[i])):
                f_write.write(list_atoms[all_atoms[i][j]] + " ")
                #print(list_atoms[all_atoms[i][j]])
            #print("]")
            f_write.write("]")
        f_write.write("]")
        #print("]")

        #########################################
        #Printing atomic types
        #########################################
        f_write.write("\n[")
        #print("[")
        for i in range(len(all_atom_types)):
            #print("[")
            f_write.write("[")
            for j in range(len(all_atom_types[i])):
                f_write.write(list_atoms_types[all_atom_types[i][j]] + " ")
                #print(list_atoms_types[all_atom_types[i][j]])
            #print("]")
            f_write.write("]")
        f_write.write("]")
        #print("]")
        
        #########################################
        #Printing secondary structure
        #########################################
        model = structure[0]
        dssp = DSSP(model, file_data_name, dssp='mkdssp')
        secondary_struct = ""
        for i in range(len(list(dssp.keys()))):
            if dssp.keys()[i][0] != num_chain:
                continue
            a_key = list(dssp.keys())[i]
            secondary_struct += dssp[a_key][2]
        f_write.write("\n" + secondary_struct)
        num_of_structures+=1
        f_write.write("\n\n")
        os.remove(new_name)


        
