
from simtk import unit, openmm
from simtk.openmm import app
import numpy as np
from sys import stdout
from openmmtools import integrators


pdb = app.PDBFile('ala2_2ndband.pdb')
topology = pdb.getTopology()
positions = pdb.getPositions()


# ff = app.ForceField('amber99sbildn.xml',"amber96_obc.xml")
# system = ff.createSystem(
#     topology=topology, 
#     constraints=app.HBonds, 
#     rigidWater=True
#     )

with open('ala2_noconstraints_system.txt') as f:
    xml = f.read()
system = openmm.XmlSerializer.deserialize(xml)

platform = openmm.Platform.getPlatform(2)

temperature = 300.0 * unit.kelvin
collision_rate = 1.0 / unit.picosecond
timestep = 1.0 * unit.femtosecond
reportInterval = 1000
steps = 1E+8
fname = 'test'
parametersdict = {'Collision rate':collision_rate,'Temperature':temperature,'Timestep':timestep,'Report Interval':reportInterval}
import pickle
f_p = open(f'parameters/parameters{fname}.pkl','wb')
pickle.dump(parametersdict,f_p)
f_p.close

integrator = integrators.LangevinIntegrator(temperature=temperature,collision_rate=collision_rate,timestep=timestep)
properties_dict = {}
properties_dict["DeviceIndex"] = "3"
simulation = app.Simulation(topology, system, integrator,platform,platformProperties=properties_dict)
simulation.context.setPositions(positions)
simulation.minimizeEnergy()
simulation.context.setVelocitiesToTemperature(temperature)
simulation.reporters.append(app.StateDataReporter(stdout, reportInterval=reportInterval*10, step=True, potentialEnergy=True,temperature=True))
simulation.reporters.append(app.DCDReporter(f'Trajectories/{fname}.dcd',reportInterval))
simulation.reporters.append(openmm.app.StateDataReporter(
    f'Trajectories/{fname}.txt', 
    reportInterval=reportInterval, 
    step=True, 
    potentialEnergy=True,
    kineticEnergy = True,
    totalEnergy = True,
    temperature=True,
))

simulation.step(steps)

