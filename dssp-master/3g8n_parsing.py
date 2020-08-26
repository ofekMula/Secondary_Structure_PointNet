from Bio.PDB import *
from Bio.PDB.DSSP import DSSP

print("START")
p = PDBParser()
structure = p.get_structure("X", "3g8n.pdb")
model = structure[0]

#chain = model['A']
#print("RESIDES #:", len(list(chain)))

dssp = DSSP(model, "3g8n.pdb", dssp = 'mkdssp')
print(dssp.keys()[2][1][1]) #RESIDUE NUMBER

result = ""
for i in range(len(list(dssp.keys()))):
	if dssp.keys()[i][0] != 'A':
		continue
	a_key = list(dssp.keys())[i]
	result+=dssp[a_key][2]
print("STRING", result)
print(len(result))

#a_key = list(dssp.keys())[4]
#print("RESULT", dssp[a_key])
#print("LEN", len(list(dssp.keys())))
