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




        
