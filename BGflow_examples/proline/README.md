## Description of files

cis_openmm_script.py: Python script to run implicit solvent MD of capped proline - starting from cis conformation\
cis_pro_xmlsystem.txt: OpenMM system file with bond constraints\
cis_pro.pdb: PDB file of cis capped proline\
noconstraints_xmlsystem.txt: OpenMM system file with no bond constraints\
pro_BG_GPU1.py: Python script for training a BG - requires a MD training dataset (pro_openmm.ipynb)\
pro_openmm.ipynb: Notebook to run implicit solvent MD of capped proline\
proline_BG.ipynb: Notebook for training a BG - requires a MD training dataset (pro_openmm.ipynb)\
trans_openmm_script.py: Python script to run implicit solvent MD of capped proline - starting from trans conformation\
trans_pro_reindexed.pdb: PDB file of trans capped proline, atom indexing matches cis PDB file\
trans_pro.pdb: PDB file of trans capped proline, atom indexing DOES NOT match cis PDB file - topologies don't match
