import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)
import Protein_Utils.py

protein_paths = [line.rstrip() for line in open(os.path.join(BASE_DIR, 'meta/protein_paths.txt'))]
protein_paths = [os.path.join(Protein_Utils.DATA_PATH, p) for p in protein_paths]

output_folder = os.path.join(ROOT_DIR, 'data/protein_output_collect')
if not os.path.exists(output_folder):
    os.mkdir(output_folder)

for protein_path in protein_paths:
    print(protein_path)
    try:
        elements = protein_path.split('/')
        out_filename = elements[-3]+'_'+elements[-2]+'.npy'  # Area_1_11as_A.npy
        Protein_Utils.collect_point_label(protein_path, os.path.join(output_folder, out_filename), 'numpy')
    except:
        print(protein_path, 'ERROR!!')
