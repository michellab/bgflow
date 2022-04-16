# Folder for scripts using the bgflow repository to couple Boltzmann generators with molecular dynamics
To reproduce the coupled scheme, there are 3 stages:
1. Create a training dataset - MD simulation with openmm
2. Train a Boltzmann generator 
3. Run the coupled scheme

Scripts for the first two stages are inside the Alanine_dipeptide and proline folders
## Notebooks
coupled_BGandMD_ala2.ipynb: Coupled sampling scheme for alanine dipeptide - requires trained generator (Alanine_dipeptide/alanine_dipeptide_basics.ipynb)\
coupled_BGandMD_proline.ipynb: Coupled sampling scheme for capped proline - requires trained generator (proline/proline_BG.ipynb)\
Decomposing_energies.ipynb: Plotting script for breakdown by energy component\
plotting_energies.ipynb: Plotting script for energy distributions from trajectory files\
plotting_probabilities.ipynb: Plotting script for the acceptance probability breakdown from the coupled scheme\
Ramachandranplot.ipynb: Plotting script for Ramachandran plots\
VMDhistograms.ipynb: Plotting script for internal coordinate distributions of trajectories

Coupled scheme, Boltzmann generator and MD simulation scripts also have .py files
