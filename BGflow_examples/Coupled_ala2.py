
from simtk import unit
from simtk import openmm

import numpy as np
from sys import stdout
from openmmtools import integrators
import random
import matplotlib.pyplot as plt
import csv

# pdb = app.PDBFile('ala2_fromURL.pdb')
# topology = pdb.getTopology()
# positions = pdb.getPositions(asNumpy=True).value_in_unit(unit.nanometer)

# ff = app.ForceField('amber99sbildn.xml',"amber96_obc.xml")
# system = ff.createSystem(
#     topology=topology, 
#     removeCMMotion=True,
#     nonbondedMethod=app.NoCutoff,
#     constraints=app.HBonds, 
#     rigidWater=True
#     )

with open('Alanine_dipeptide/ala2_noconstraints_system.txt') as f:
    xml = f.read()
noconstr_system = openmm.XmlSerializer.deserialize(xml)
#platform 2 = CUDA
platform = openmm.Platform.getPlatform(2)

temperature_bg = 300.0 * unit.kelvin
collision_rate_bg = 1.0 / unit.picosecond
timestep_bg = 4.0 * unit.femtosecond


#Setting up generator
import torch

device = "cuda:1" if torch.cuda.is_available() else "cpu"
dtype = torch.float32
# a context tensor to send data to the right device and dtype via '.to(ctx)'
ctx = torch.zeros([], device=device, dtype=dtype)

#need to load a dataset for dimensions of BG and to set up Mixed Coordinate Transform which requires data as an argument
import mdtraj
rough_dataset = mdtraj.load('Alanine_dipeptide/Trajectories/300K.dcd', top='Alanine_dipeptide/ala2_fromURL.pdb')
reference_frame = rough_dataset[0]
dataset = rough_dataset.superpose(reference_frame)

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

#system setup, probably need to write a function to do this
# from simtk import openmm
# with open('ala2_xml_system.txt') as f:
#     xml = f.read()
# system = openmm.XmlSerializer.deserialize(xml)
from bgflow.distribution.energy.openmm import OpenMMBridge, OpenMMEnergy

integrator = integrators.LangevinIntegrator(temperature=temperature_bg,collision_rate=collision_rate_bg,timestep=timestep_bg)
energy_bridge = OpenMMBridge(noconstr_system, integrator, n_workers=1)
target_energy = OpenMMEnergy(int(dim), energy_bridge)

#setting up training_data argument for MixedCoordinateTransform - not sure how much effect this has
n_train = len(dataset)//2
n_test = len(dataset) - n_train
permutation = np.random.permutation(n_train)
all_data = dataset.xyz.reshape(-1, dimensions(dataset))
training_data = torch.tensor(all_data[permutation]).to(ctx)
test_data = torch.tensor(all_data[permutation + n_train]).to(ctx)

import bgflow as bg

dim_cartesian = len(rigid_block) * 3 - 6
dim_bonds = len(z_matrix)
dim_angles = dim_bonds
dim_torsions = dim_bonds

#set up coordinate transform layer
coordinate_transform = bg.MixedCoordinateTransformation(
    data=training_data, 
    z_matrix=z_matrix,
    fixed_atoms=rigid_block,
    keepdims=dim_cartesian, 
    normalize_angles=True,
).to(ctx)

#setting up prior distribution
dim_ics = dim_bonds + dim_angles + dim_torsions + dim_cartesian
mean = torch.zeros(dim_ics).to(ctx) 
# passing the mean explicitly to create samples on the correct device
prior = bg.NormalDistribution(dim_ics, mean=mean)

split_into_ics_flow = bg.SplitFlow(dim_bonds, dim_angles, dim_torsions, dim_cartesian)

#defining RealNVP
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

#setting up normalising flow composed of RealNVP followed by coordinate transform
n_realnvp_blocks = 5
layers = []

for i in range(n_realnvp_blocks):
    layers.append(RealNVP(dim_ics, hidden=[128, 128, 128]))
layers.append(split_into_ics_flow)
layers.append(bg.InverseFlow(coordinate_transform))

flow = bg.SequentialFlow(layers).to(ctx)

#loading trained model into empty
flow.load_state_dict(torch.load('Alanine_dipeptide/Models/model300K_noconstr_long_1.pt'))

#setting up generator
generator = bg.BoltzmannGenerator(
    flow=flow,
    prior=prior,
    target=target_energy)


def getbg_positions(n_atoms):    
    bg_positions_tensor, dlogp_tensor = generator.sample(1,with_dlogp=True)
    bg_positions = bg_positions_tensor.cpu().detach().numpy().reshape(n_atoms,3)
    dlogp = dlogp_tensor.cpu().detach().numpy()
    #return bg_positions, np.exp(np.abs(dlogp))
    #return bg_positions, dlogp
    return bg_positions, np.abs(dlogp[0,0])


def getbias(positions,n_atoms):
    torch_positions = torch.tensor(positions.reshape(-1,n_atoms*3)).to(ctx)
    z, dlogp_inverse_tensor = flow.forward(torch_positions,inverse=True)
    dlogp_inverse = dlogp_inverse_tensor.cpu().detach().numpy()
    #return np.exp(np.abs(dlogp_inverse))
    return np.abs(dlogp_inverse[0,0])
    #return -dlogp_inverse


def getthermalenergy(temperature):
    #unit.BOLTZMANN_CONSTANT_kB is in units of J/K
    kb = unit.BOLTZMANN_CONSTANT_kB * unit.AVOGADRO_CONSTANT_NA
    kt = kb * temperature
    kt = kt.value_in_unit(unit.kilojoule_per_mole)
    beta = 1/kt
    return beta



cycles = 1000
MDsteps = 1E6
BGmoves = 1000
MDreport_interval = 10000

fname = 'coupled_1nsMD_1000cycles_1000BGtrials'


##Setting up MD and initialising

pdb = openmm.app.PDBFile('Alanine_dipeptide/ala2_fromURL.pdb')
topology = pdb.getTopology()
positions = pdb.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
n_atoms = positions.shape[0]

md_temperature = 300 * unit.kelvin
md_collision_rate = 10 / unit.picosecond
md_timestep = 1.0 * unit.femtosecond

parametersdict = {'Collision rate':md_collision_rate,'Temperature':md_temperature,'Timestep':md_timestep}
import pickle
with open(f'Alanine_dipeptide/parameters/parameters{fname}.pkl','wb') as f_p:
    pickle.dump(parametersdict,f_p)
f_p.close

integrator = integrators.LangevinIntegrator(temperature=md_temperature,collision_rate=md_collision_rate,timestep=md_timestep)
#integrator.setConstraintTolerance(0.00001)
#integrator = openmm.VerletIntegrator(timestep)
properties_dict = {}
properties_dict["DeviceIndex"] = "2"
simulation = openmm.app.Simulation(topology, noconstr_system, integrator,platform,platformProperties=properties_dict)
simulation.context.setPositions(positions)
simulation.minimizeEnergy()
simulation.context.setVelocitiesToTemperature(md_temperature)
simulation.step(1000)


##WITH BIAS

#reporter for details about states - kinetic energy etc.
textfile_reporter = openmm.app.StateDataReporter(
    f'Coupled_scheme/Trajectories/{fname}.txt', 
    reportInterval=MDreport_interval, 
    step=True, 
    potentialEnergy=True,
    kineticEnergy = True,
    totalEnergy = True,
    temperature=True,
)

simulation.reporters.append(textfile_reporter)

#reporter to watch progress of simulation - to stdout
simulation.reporters.append(openmm.app.StateDataReporter(
    stdout, 
    reportInterval=MDreport_interval, 
    step=True, 
    totalEnergy = True,
    temperature=True,
))

#reporter for trajectory - positions
trajectory_reporter = openmm.app.DCDReporter(f'Coupled_scheme/Trajectories/{fname}.dcd',reportInterval=MDreport_interval)
simulation.reporters.append(trajectory_reporter)

accept_counter = 0
acceptance_probs = []
biases = []
bias_ratios = []
energy_terms = []

for x in range(cycles):
    print('cycle',x)
    acceptance = False
    simulation.step(MDsteps)
    current_state = simulation.context.getState(getEnergy=True,getPositions=True)
    current_positions = current_state.getPositions(asNumpy=True)
    current_positions_astraj = mdtraj.Trajectory(xyz=current_positions, topology=mdtraj.load('Alanine_dipeptide/ala2_fromURL.pdb').topology)
    current_positions_aligned = current_positions_astraj.superpose(reference_frame).xyz
    bias_current = getbias(current_positions_aligned,n_atoms=n_atoms)

    current_total_energy = current_state.getKineticEnergy() + current_state.getPotentialEnergy()
    print('MD_end_energy',current_total_energy)
    for y in range(BGmoves):  
        integrator = integrators.LangevinIntegrator(temperature=md_temperature,collision_rate=md_collision_rate,timestep=md_timestep)
        bgsimulation = openmm.app.Simulation(topology,noconstr_system,integrator,platform,platformProperties=properties_dict)
        bg_positions, bias_new = getbg_positions(n_atoms=n_atoms)
        bgsimulation.context.setPositions(bg_positions)
        bgsimulation.context.setVelocitiesToTemperature(md_temperature)
        new_state = bgsimulation.context.getState(getEnergy=True,getPositions=True)
        new_total_energy = new_state.getKineticEnergy() + new_state.getPotentialEnergy()
        energy_change = (new_total_energy - current_total_energy).value_in_unit(unit.kilojoule_per_mole)
        bias_change = bias_current-bias_new
        #bias_ratios.append(np.exp(bias_change)[0,0])
        #energy_terms.append(np.exp(-getthermalenergy(md_temperature)*energy_change))

        acceptance_prob = min(1,(np.exp(bias_change-getthermalenergy(md_temperature)*energy_change)))
        #acceptance_prob = min(1,(np.exp(-getthermalenergy(md_temperature)*energy_change)))#*bias_current/bias_new))
        #acceptance_probs.append(acceptance_prob)

        f = open(f"Coupled_scheme/probability_files/{fname}_prob_breakdown.csv", "a")
        writer = csv.writer(f)
        writer.writerow((acceptance_prob, np.exp(bias_change), np.exp(-getthermalenergy(md_temperature)*energy_change)))
        f.close()  

        random_val = random.random()
        # if random_val < acceptance_prob:
        #     if acceptance == False:
        #         print('accept new conformation')
        #         print('accepted BG energy',new_total_energy)
        #         new_checkpoint = bgsimulation.context.createCheckpoint()
        #         simulation.context.loadCheckpoint(new_checkpoint)
        #         BG_state = simulation.context.getState(getEnergy=True,getPositions=True)
        #         trajectory_reporter.report(simulation,BG_state)
        #         textfile_reporter.report(simulation,BG_state)
        #         #accept_counter += 1
        #         acceptance = True
        #     #break
        # else:
        #     print('rejected BG energy',y,new_total_energy)

        if random_val < acceptance_prob:
            print('accept new conformation')
            print('accepted BG energy',new_total_energy)
            new_checkpoint = bgsimulation.context.createCheckpoint()
            simulation.context.loadCheckpoint(new_checkpoint)
            BG_state = simulation.context.getState(getEnergy=True,getPositions=True)
            trajectory_reporter.report(simulation,BG_state)
            textfile_reporter.report(simulation,BG_state)
            #accept_counter += 1
            #acceptance = True
            break
        else:
            print('rejected BG energy',y,new_total_energy)
       


