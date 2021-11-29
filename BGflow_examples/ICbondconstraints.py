import sys
import bgflow as bg
import torch

#from bgflow.distribution.energy.openmm import OpenMMBridge

device = "cuda:1" if torch.cuda.is_available() else "cpu"
dtype = torch.float32

# a context tensor to send data to the right device and dtype via '.to(ctx)'
ctx = torch.zeros([], device=device, dtype=dtype)

import mdtraj
#dataset = mdtraj.load('output.dcd', top='ala2_fromURL.pdb')
dataset = mdtraj.load('BGflow_examples/TSFtraj.dcd', top='BGflow_examples/ala2_fromURL.pdb', stride=4)
#fname is file title for saving figures - not the full path
#fname = "obcimplicit_stride10_repeat2"
coordinates = dataset.xyz
from bgmol.datasets import Ala2TSF300
target_energy = Ala2TSF300().get_energy_model(n_workers=1)
#energy_bridge = OpenMMBridge()

#rigid_block indicates which atoms are described in cartesian coordinates
#z_matrix gives an set of atom indices to describe each atom in terms of a bond (column 1), angle (column 2) and dihedral (column 3)
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

#split into training and testing set
n_train = len(dataset)//2
n_test = len(dataset) - n_train
permutation = np.random.permutation(n_train)
all_data = coordinates.reshape(-1, dimensions(dataset))
training_data = torch.tensor(all_data[permutation]).to(ctx)
test_data = torch.tensor(all_data[permutation + n_train]).to(ctx)

#defining dimensions for the prior distribution
dim_cartesian = len(rigid_block) * 3 - 6

dim_bonds = len(z_matrix) - 9
dim_angles = len(z_matrix)
dim_torsions = len(z_matrix)

#print(torch.full((2, 3), 3.141592))

#mixed coordinate transformation, rigid block undergoes PCA and all other atoms are defined as internal coords 
coordinate_transform = bg.MixedCoordinateTransformation(
    data=training_data, 
    z_matrix=z_matrix,
    fixed_atoms=rigid_block,
    #keepdims=None,
    keepdims=dim_cartesian, 
    normalize_angles=True,
).to(ctx)

#testing coordinate transform
#bonds, angles, torsions, cartesian, dlogp = coordinate_transform.forward(training_data[:3])
#x, dlogp = coordinate_transform.forward(bonds, angles, torsions, cartesian, inverse=True)
#print(x, dlogp)

#defining the prior distribution - multidimensional Gaussian
dim_ics = dim_bonds + dim_angles + dim_torsions + dim_cartesian
mean = torch.zeros(dim_ics).to(ctx) 
# passing the mean explicitly to create samples on the correct device
prior = bg.NormalDistribution(dim_ics, mean=mean)

split_into_ics_flow = bg.SplitFlow(dim_bonds, dim_angles, dim_torsions, dim_cartesian)

# test
_ics = split_into_ics_flow(prior.sample(3))[:-1]
x = coordinate_transform.forward(*_ics, inverse=True)[0].shape


#defining the RealNVP normalising flow block
class RealNVP(bg.SequentialFlow):
    
    def __init__(self, dim, hidden):
        self.dim = dim
        self.hidden = hidden
        super().__init__(self._create_layers())
    
    def _create_layers(self):
        
        dim_channel1 =  self.dim//2
        dim_channel2 = self.dim - dim_channel1
        print('ch1',dim_channel1)
        print('ch2',dim_channel2)
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
        print('dim1',dim1)
        print('dim2',dim2)
        return bg.DenseNet(
            [dim1, *self.hidden, dim2],
            activation=torch.nn.ReLU()
        )
    
    def _coupling_block(self, dim1, dim2):
        return bg.CouplingFlow(bg.AffineTransformer(
            shift_transformation=self._dense_net(dim1, dim2),
            scale_transformation=self._dense_net(dim1, dim2)
        ))
    
#testing RealNVP
#RealNVP(dim_ics, hidden=[128]).to(ctx).forward(prior.sample(3))[0].shape

#forming the whole flow: 5 NVP blocks, then split into the right dimensions 
#and then inverse coordinate transform (ICs to cartesian)
n_realnvp_blocks = 5
layers = []

for i in range(n_realnvp_blocks):
    layers.append(RealNVP(dim_ics, hidden=[128, 128, 128]))
layers.append(split_into_ics_flow)
layers.append(bg.InverseFlow(coordinate_transform))

flow = bg.SequentialFlow(layers).to(ctx)

# test
x = flow.forward(prior.sample(3))
print(flow.forward(x[0], inverse=True))

print('test done')

generator = bg.BoltzmannGenerator(
    flow=flow,
    prior=prior,
    target=target_energy
)

nll_optimizer = torch.optim.Adam(generator.parameters(), lr=1e-3)
nll_trainer = bg.KLTrainer(
    generator, 
    optim=nll_optimizer,
    train_energy=False
)

nll_trainer.train(
        n_iter=1, 
        data=training_data,
        batchsize=10,
        n_print=1, 
        w_energy=None
    )

mixed_optimizer = torch.optim.Adam(generator.parameters(), lr=1e-4)
mixed_trainer = bg.KLTrainer(
    generator, 
    optim=mixed_optimizer,
    train_energy=True
)

mixed_trainer.train(
    n_iter=3, 
    data=training_data,
    batchsize=10,
    n_print=1, 
    w_energy=0.1,
    w_likelihood=0.9,
    clip_forces=20.0
    )

#torch.save(flow.state_dict(),f'TSFstride4_cmdline.pt')

samples = generator.sample(10000)

samplestrajectory = mdtraj.Trajectory(
            xyz=samples.cpu().detach().numpy().reshape(-1, 22, 3), 
            topology=mdtraj.load('BGflow_examples/ala2_fromURL.pdb').topology
)
#print(samples)
#samplestrajectory.save('BGflow_examples/constr_samples/newconstr_samples_TSF.dcd')

sys.exit()