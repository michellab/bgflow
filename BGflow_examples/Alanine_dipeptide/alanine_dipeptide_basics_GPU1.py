
import torch

device = "cuda:1" if torch.cuda.is_available() else "cpu"
dtype = torch.float32

# a context tensor to send data to the right device and dtype via '.to(ctx)'
ctx = torch.zeros([], device=device, dtype=dtype)
# a brief check if this module is the main executable (or imported)

import mdtraj
#dataset = mdtraj.load('output.dcd', top='ala2_fromURL.pdb')
rough_dataset = mdtraj.load('TSFtraj.dcd', top='ala2_fromURL.pdb')
dataset = rough_dataset.superpose(rough_dataset[0])
fname = "TSFtraj_20000KLL_1"

coordinates = dataset.xyz

from simtk import unit
temperature = 300.0 * unit.kelvin
collision_rate = 1.0 / unit.picosecond
timestep = 4.0 * unit.femtosecond

n_iter_NLL = 20000
n_iter_mixed = 20000

import numpy as np
rigid_block = np.array([6, 8, 9, 10, 14])
z_matrix = np.array([
    [0, 1, 4, 6],
    [1, 4, 6, 8],
    [2, 1, 4, 0],
    [3, 1, 4, 0],
    [4, 6, 8, 14],
    [5, 4, 6, 8],
    [7, 6, 8, 4],
    [11, 10, 8, 6],
    [12, 10, 8, 11],
    [13, 10, 8, 11],
    [15, 14, 8, 16],
    [16, 14, 8, 6],
    [17, 16, 14, 15],
    [18, 16, 14, 8],
    [19, 18, 16, 14],
    [20, 18, 16, 19],
    [21, 18, 16, 19]
])


def dimensions(dataset):
        return np.prod(dataset.xyz[0].shape)
dim = dimensions(dataset)
print(dim)


from simtk import openmm
with open('ala2_xml_system.txt') as f:
    xml = f.read()
system = openmm.XmlSerializer.deserialize(xml)

from bgflow.distribution.energy.openmm import OpenMMBridge, OpenMMEnergy
from openmmtools import integrators

integrator = integrators.LangevinIntegrator(temperature=temperature,collision_rate=collision_rate,timestep=timestep)

energy_bridge = OpenMMBridge(system, integrator, n_workers=1)
target_energy = OpenMMEnergy(int(dim), energy_bridge)

def compute_phi_psi(trajectory):
    phi_atoms = [4, 6, 8, 14]
    phi = md.compute_dihedrals(trajectory, indices=[phi_atoms])[:, 0]
    psi_atoms = [6, 8, 14, 16]
    psi = md.compute_dihedrals(trajectory, indices=[psi_atoms])[:, 0]
    return phi, psi

import numpy as np
import mdtraj as md 
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm

def plot_phi_psi(ax, trajectory):
    if not isinstance(trajectory, md.Trajectory):
        trajectory = md.Trajectory(
            xyz=trajectory.cpu().detach().numpy().reshape(-1, 22, 3), 
            topology=md.load('ala2_fromURL.pdb').topology
        )
    phi, psi = compute_phi_psi(trajectory)
    
    ax.hist2d(phi, psi, 50, norm=LogNorm())
    ax.set_xlim(-np.pi, np.pi)
    ax.set_ylim(-np.pi, np.pi)
    ax.set_xlabel("$\phi$")
    _ = ax.set_ylabel("$\psi$")
    
    return trajectory


n_train = len(dataset)//2
n_test = len(dataset) - n_train
permutation = np.random.permutation(n_train)

all_data = coordinates.reshape(-1, dimensions(dataset))
training_data = torch.tensor(all_data[permutation]).to(ctx)
test_data = torch.tensor(all_data[permutation + n_train]).to(ctx)

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

bonds, angles, torsions, cartesian, dlogp = coordinate_transform.forward(training_data[:3])
bonds.shape, angles.shape, torsions.shape, cartesian.shape, dlogp.shape
#print(bonds)

dim_ics = dim_bonds + dim_angles + dim_torsions + dim_cartesian
mean = torch.zeros(dim_ics).to(ctx) 
# passing the mean explicitly to create samples on the correct device
prior = bg.NormalDistribution(dim_ics, mean=mean)

split_into_ics_flow = bg.SplitFlow(dim_bonds, dim_angles, dim_torsions, dim_cartesian)


# test
#print(prior.sample(3))
_ics = split_into_ics_flow(prior.sample(1))
#print(_ics)
coordinate_transform.forward(*_ics, inverse=True)[0].shape

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

# test
flow.forward(prior.sample(3))[0].shape

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

mixed_optimizer = torch.optim.Adam(generator.parameters(), lr=1e-4)
mixed_trainer = bg.KLTrainer(
    generator, 
    optim=mixed_optimizer,
    train_energy=True
)

mixed_trainer.train(
    n_iter=n_iter_mixed, 
    data=training_data,
    batchsize=1000,
    n_print=100, 
    w_energy=0.1,
    w_likelihood=0.9,
    clip_forces=20.0
    )

torch.save(flow.state_dict(),f'model{fname}.pt')
#torch.save(generator.state_dict(keep_vars=True),f'model{fname}.pt')

n_samples = 10000
samples = generator.sample(n_samples)
print(samples)

fig, axes = plt.subplots(1, 2, figsize=(6,3))
fig.tight_layout()

samplestrajectory = plot_phi_psi(axes[0], samples)
samplestrajectory.save(f"{fname}_samplestraj.dcd")
#plot_energies(axes[1], samples, target_energy, test_data)
plt.savefig(f"varysnapshots/{fname}.png", bbox_inches = 'tight')