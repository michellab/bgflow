

import torch

device = "cuda:1" if torch.cuda.is_available() else "cpu"
dtype = torch.float32

# a context tensor to send data to the right device and dtype via '.to(ctx)'
ctx = torch.zeros([], device=device, dtype=dtype)

import os
import mdtraj
rough_dataset_trans = mdtraj.load('Trajectories/trans_pro_300K_noconstr_long.dcd', top='trans_pro_reindexed.pdb')
rough_dataset_cis = mdtraj.load('Trajectories/cis_pro_300K_noconstr_long.dcd',top='cis_pro.pdb')

rough_dataset = rough_dataset_trans.join(rough_dataset_cis)

dataset = rough_dataset.superpose(rough_dataset_trans[0])
#dataset.save('pro_joined.dcd')

md_fname = 'trans_pro_300K_noconstr_long'
fname = "unconstrainedMD_300K_10blocks_1"

coordinates = dataset.xyz
n_atoms = 26

from simtk import openmm, unit
import pickle
try:
    pickleFile = open(f'parameters/parameters{md_fname}.pkl','rb')
    parametersdict = pickle.load(pickleFile)
    temperature = parametersdict['Temperature']
    collision_rate = parametersdict['Collision rate']
    timestep = parametersdict['Timestep']
    reportInterval = parametersdict['Report Interval']
except:
    print('no parameters found')
    temperature = 300.0 * unit.kelvin
    collision_rate = 1.0 / unit.picosecond
    timestep = 1.0 * unit.femtosecond



n_iter_NLL = 20000
n_iter_mixed = 20000


import numpy as np
rigid_block = np.array([9,7,6])
z_matrix = np.array([
    [0,4,6,7],
    [1,0,4,6],
    [2,0,4,6],
    [3,0,4,6],
    [4,6,7,9],
    [5,4,6,7],
    [8,7,6,4],
    [10,9,7,6],
    [11,7,9,24],
    [12,11,7,6],
    [13,11,7,6],
    [14,17,6,4],
    [15,14,17,6],
    [16,14,17,6],
    [17,6,7,9],
    [18,17,6,7],
    [19,17,6,7],
    [20,24,9,7],
    [21,20,24,9],
    [22,20,24,9],
    [23,20,24,9],
    [24,9,7,6],
    [25,24,9,7]
])




def dimensions(dataset):
        return np.prod(dataset.xyz[0].shape)
dim = dimensions(dataset)


from simtk import openmm
with open('noconstraints_xmlsystem.txt') as f:
    xml = f.read()
system = openmm.XmlSerializer.deserialize(xml)

from bgflow.distribution.energy.openmm import OpenMMBridge, OpenMMEnergy
from openmmtools import integrators

integrator = integrators.LangevinIntegrator(temperature=temperature,collision_rate=collision_rate,timestep=timestep)

energy_bridge = OpenMMBridge(system, integrator, n_workers=1)
target_energy = OpenMMEnergy(int(dim), energy_bridge)


import mdtraj as md 


def compute_phi_psi(trajectory):
    phi_atoms = [4, 6, 7,9]
    phi = md.compute_dihedrals(trajectory, indices=[phi_atoms])[:, 0]
    psi_atoms = [6, 7, 9, 24]
    psi = md.compute_dihedrals(trajectory, indices=[psi_atoms])[:, 0]
    return phi, psi


import numpy as np
import mdtraj as md 
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm

def plot_phi_psi(ax, trajectory):
    if not isinstance(trajectory, md.Trajectory):
        trajectory = md.Trajectory(
            xyz=trajectory.cpu().detach().numpy().reshape(-1, n_atoms, 3), 
            topology=md.load('cis_pro.pdb').topology
        )
    phi, psi = compute_phi_psi(trajectory)
    
    ax.hist2d(phi, psi, 50, norm=LogNorm())
    ax.set_xlim(-np.pi, np.pi)
    ax.set_ylim(-np.pi, np.pi)
    ax.set_xlabel("$\phi$")
    _ = ax.set_ylabel("$\psi$")
    
    return trajectory

def plot_omega(ax, trajectory):

    if not isinstance(trajectory, md.Trajectory):
        trajectory = md.Trajectory(
            xyz=trajectory.cpu().detach().numpy().reshape(-1, n_atoms, 3), 
            topology=md.load('cis_pro.pdb').topology
        )
    
    omega_atoms = [0,4,6,7]
    omega = md.compute_dihedrals(trajectory, indices=[omega_atoms])[:, 0]
    ax.hist(omega, bins=40)
    ax.set_xlim(-np.pi, np.pi)
    ax.set_xlabel(f"$\omega$")
    ax.set_ylabel(f"Count   [#Samples / {len(omega)}]")


fig, axes = plt.subplots(1,2,figsize=(6,3))
trajectory_md = plot_phi_psi(axes[0], dataset)
plot_omega(axes[1],dataset)
plt.savefig(f'BG_training_plots/datasetRplots/Rplot{fname}.png',bbox_inches='tight')

# ## Split Data and Randomly Permute Samples


n_train = len(dataset)//2
n_test = len(dataset) - n_train
permutation = np.random.permutation(n_train)

all_data = coordinates.reshape(-1, dimensions(dataset))
np.random.shuffle(all_data)
training_data = torch.tensor(all_data[permutation]).to(ctx)
test_data = torch.tensor(all_data[permutation + n_train]).to(ctx)


# ## Define the Internal Coordinate Transform
# 
# Rather than generating all-Cartesian coordinates, we use a mixed internal coordinate transform.
# The five central alanine atoms will serve as a Cartesian "anchor", from which all other atoms are placed with respect to internal coordinates (IC) defined through a z-matrix. We have deposited a valid `z_matrix` and the corresponding `rigid_block` in the `dataset.system` from `bgmol`.


import bgflow as bg


# throw away 6 degrees of freedom (rotation and translation)
dim_cartesian = len(rigid_block) * 3 - 6
print(dim_cartesian)
#dim_cartesian = len(system.rigid_block) * 3
dim_bonds = len(z_matrix)
print(dim_bonds)
dim_angles = dim_bonds
dim_torsions = dim_bonds


coordinate_transform = bg.MixedCoordinateTransformation(
    data=training_data, 
    z_matrix=z_matrix,
    fixed_atoms=rigid_block,
    #keepdims=None,
    keepdims=dim_cartesian, 
    normalize_angles=True,
).to(ctx)


# For demonstration, we transform the first 3 samples from the training data set into internal coordinates as follows:


bonds, angles, torsions, cartesian, dlogp = coordinate_transform.forward(training_data[:3])
bonds.shape, angles.shape, torsions.shape, cartesian.shape, dlogp.shape
#print(bonds)


# ## Prior Distribution
# 
# The next step is to define a prior distribution that we can easily sample from. The normalizing flow will be trained to transform such latent samples into molecular coordinates. Here, we just take a normal distribution, which is a rather naive choice for reasons that will be discussed in other notebooks.


dim_ics = dim_bonds + dim_angles + dim_torsions + dim_cartesian
mean = torch.zeros(dim_ics).to(ctx) 
# passing the mean explicitly to create samples on the correct device
prior = bg.NormalDistribution(dim_ics, mean=mean)
print(prior.sample(1).size())


# ## Normalizing Flow
# 
# Next, we set up the normalizing flow by stacking together different neural networks. For now, we will do this in a rather naive way, not distinguishing between bonds, angles, and torsions. Therefore, we will first define a flow that splits the output from the prior into the different IC terms.
# 
# ### Split Layer


split_into_ics_flow = bg.SplitFlow(dim_bonds, dim_angles, dim_torsions, dim_cartesian)


# test
#print(prior.sample(3))
ics = split_into_ics_flow(prior.sample(1))
#print(ics)
coordinate_transform.forward(*ics, inverse=True)


# ### Coupling Layers
# 
# Next, we will set up so-called RealNVP coupling layers, which split the input into two channels and then learn affine transformations of channel 1 conditioned on channel 2. Here we will do the split naively between the first and second half of the degrees of freedom.


class RealNVP(bg.SequentialFlow):
    
    def __init__(self, dim, hidden):
        self.dim = dim
        self.hidden = hidden
        super().__init__(self._create_layers())
    
    def _create_layers(self):
        dim_channel1 =  self.dim//2
        dim_channel2 = self.dim - dim_channel1
        split_into_2 = bg.SplitFlow(dim_channel1, dim_channel2)
        
        layers = [
            # -- split
            split_into_2,
            # --transform
            self._coupling_block(dim_channel1, dim_channel2),
            bg.SwapFlow(),
            self._coupling_block(dim_channel2, dim_channel1),
            # -- merge
            bg.InverseFlow(split_into_2)
        ]
        return layers
        
    def _dense_net(self, dim1, dim2):
        return bg.DenseNet(
            [dim1, *self.hidden, dim2],
            activation=torch.nn.ReLU()
        )
    
    def _coupling_block(self, dim1, dim2):
        return bg.CouplingFlow(bg.AffineTransformer(
            shift_transformation=self._dense_net(dim1, dim2),
            scale_transformation=self._dense_net(dim1, dim2)
        ))
    


RealNVP(dim_ics, hidden=[128]).to(ctx).forward(prior.sample(3))[0].shape


# ### Boltzmann Generator
# 
# Finally, we define the Boltzmann generator.
# It will sample molecular conformations by 
# 
# 1. sampling in latent space from the normal prior distribution,
# 2. transforming the samples into a more complication distribution through a number of RealNVP blocks (the parameters of these blocks will be subject to optimization),
# 3. splitting the output of the network into blocks that define the internal coordinates, and
# 4. transforming the internal coordinates into Cartesian coordinates through the inverse IC transform.


n_realnvp_blocks = 5
layers = []

for i in range(n_realnvp_blocks):
    layers.append(RealNVP(dim_ics, hidden=[128, 128, 128]))
layers.append(split_into_ics_flow)
layers.append(bg.InverseFlow(coordinate_transform))

flow = bg.SequentialFlow(layers).to(ctx)



# print number of trainable parameters
"#Parameters:", np.sum([np.prod(p.size()) for p in flow.parameters()])


generator = bg.BoltzmannGenerator(
    flow=flow,
    prior=prior,
    target=target_energy
)


def plot_energies(ax, samples, target_energy, test_data):
    sample_energies = target_energy.energy(samples).cpu().detach().numpy()
    md_energies = target_energy.energy(test_data[:len(samples)]).cpu().detach().numpy()
    cut = max(np.percentile(sample_energies, 80), 20)
    
    ax.set_xlabel("Energy   [$k_B T$]")
    # y-axis on the right
    ax2 = plt.twinx(ax)
    ax.get_yaxis().set_visible(False)
    
    ax2.hist(sample_energies, range=(-50, cut), bins=40, density=False, label="BG")
    ax2.hist(md_energies, range=(-50, cut), bins=40, density=False, label="MD")
    ax2.set_ylabel(f"Count   [#Samples / {len(samples)}]")
    ax2.legend()


# ## Train
# 
# Boltzmann generators can be trained in two ways:
# 1. by matching the density of samples from the training data via the negative log likelihood (NLL), and
# 2. by matching the target density via the backward Kullback-Leibler loss (KLL).
# 
# NLL-based training is faster, as it does not require the computation of molecular target energies. Therefore, we will first train the generator solely by density estimation.
# 
# ### NLL Training


nll_optimizer = torch.optim.Adam(generator.parameters(), lr=1e-3)
nll_trainer = bg.KLTrainer(
    generator, 
    optim=nll_optimizer,
    train_energy=False
)



nll_trainer.train(
    n_iter=n_iter_NLL, 
    data=training_data,
    batchsize=128,
    n_print=1000, 
    w_energy=0.0
)

n_samples = 10000
samples = generator.sample(n_samples)

fig, axes = plt.subplots(1, 3, figsize=(12,4))
fig.tight_layout()

plot_phi_psi(axes[0], samples)
plot_omega(axes[1],samples)
plot_energies(axes[2], samples, target_energy, test_data)
plt.tight_layout()
plt.savefig(f"BG_training_plots/{fname}NLL_learnt.png", bbox_inches="tight")

del samples

# To see what the generator has learned so far, let's first create a bunch of samples and compare their backbone angles with the molecular dynamics data. Let's also plot their energies.
# ### Mixed Training
# 
# The next step is "mixed" training with a combination of NLL and KLL. To retain some of the progress made in the NLL phase, we decrease the learning rate and increase the batch size.


mixed_optimizer = torch.optim.Adam(generator.parameters(), lr=1e-4)
mixed_trainer = bg.KLTrainer(
    generator, 
    optim=mixed_optimizer,
    train_energy=True
)


# Mixed training will be considerably slower. 
# To speed it up, you can change the settings for the OpenMM energy when creating the energy model. For example, consider not passing `n_workers=1`.
# 
# To avoid large potential energy gradients from singularities, the components of the KL gradient are constrained to (-100, 100). 



mixed_trainer.train(
    n_iter=n_iter_mixed, 
    data=training_data,
    batchsize=1000,
    n_print=100, 
    w_energy=0.1,
    w_likelihood=0.9,
    clip_forces=20.0
    )


# Plot the results:


torch.save(flow.state_dict(),f'models/model{fname}.pt')
#torch.save(generator.state_dict(keep_vars=True),f'model{fname}.pt')


n_samples = 10000
samples = generator.sample(n_samples)
#print(samples)

fig, axes = plt.subplots(1, 3, figsize=(12,4))
fig.tight_layout()

samplestrajectory = plot_phi_psi(axes[0], samples)
plot_omega(axes[1],samples)
plot_energies(axes[2], samples, target_energy, test_data)
plt.tight_layout()
plt.savefig(f"BG_training_plots/{fname}KLL_learnt.png", bbox_inches = 'tight')


samplestrajectory.save(f"Trajectories/{fname}_samplestraj.dcd")






