from Protein_Data_Base import sample_code
import os
from Bio.PDB.DSSP import DSSP
import Bio.PDB
import numpy as np
from Bio.PDB import *

##We get a protein like "1a09", a chain number like 'A'.
##That we download the pdb file & parse it.


def download_and_parse_pdb(pdb_name, num_chain):
    print ("parsing_file:", pdb_name, num_chain)
    parser = PDBParser()
    pdbl = PDBList()

    #Downloading the file
    path = "/specific/netapp5_2/iscb/wolfson/yarinluhmany/v_env/dssp-master"
    file_name = pdbl.retrieve_pdb_file(pdb_code = pdb_name, file_format = "pdb", pdir=path)
    new_name = path + '/' + pdb_name + '.pdb'
    os.rename(file_name, new_name)
    file_data_name = pdb_name+'.pdb'

    #Parsing the file
    structure = parser.get_structure(pdb_name, file_data_name)
    chain_prot = structure[0][num_chain]

    return structure, chain_prot

#create a num_py array of [x,y,z] coordinates for all atoms.
def list_of_atom_coordinates(chain_prot):
    res = sample_code.process_chain(chain_prot)
    all_coordinates = res[2]
    list_of_atoms = [np.zeros(3)]
    for i in range(len(all_coordinates)):
        for j in range(len(all_coordinates[i])):
            list_of_atoms = np.append(list_of_atoms,[all_coordinates[i][j]], axis=0)
    return np.delete(list_of_atoms, 0, axis=0)

#create a string of secondary structure for each atom.
def list_of_atom_colors(pdb_name, structure, num_chain, chain_prot):
    file_data_name = pdb_name+'.pdb'
    model = structure[0]
    dssp = DSSP(model, file_data_name, dssp='mkdssp')
    secondary_struct = ""
    for i in range(len(list(dssp.keys()))):
        if dssp.keys()[i][0]!=num_chain:
            continue
        residue_number = dssp.keys()[i][1][1]
        #print("res_num:", residue_number)
        #find relecant key in dictionary
        a_key= list(dssp.keys())[i]
        secondary_structure_of_residue = dssp[a_key][2]
        #find how many atoms in this residue
        for residue in Selection.unfold_entities(chain_prot, 'R'):
            if (sample_code.is_residue(residue)):
                if residue.get_id()[1] == residue_number:
                    num_of_atoms_in_residue = len([atom for atom in residue if sample_code.is_heavy_atom(atom)])
                    secondary_struct+=num_of_atoms_in_residue * secondary_structure_of_residue
                    break
        #add to output

    return secondary_struct

if __name__ == "__main__":
    structure, chain_prot = download_and_parse_pdb("11as", 'A')
    list_of_atoms=list_of_atom_coordinates(chain_prot)
    print(list_of_atoms, "\nlen = ", len(list_of_atoms))
    list_of_colors =list_of_atom_colors("11as", structure, 'A', chain_prot)
    print(list_of_colors, "\nlen = ", len(list_of_colors))
