from simtk import unit, openmm
from simtk.openmm import app
import numpy as np
from sys import stdout
from openmmtools import integrators
import numpy as np

from sys import stdout
import mdtraj as md
#from openmmtools import integrators


pdb = app.PDBFile('trans_pro_reindexed.pdb')

with open('noconstraints_xmlsystem.txt') as f:
    xml = f.read()
system = openmm.XmlSerializer.deserialize(xml)

## creates openmm system for passing to bg to calculate energies

# xml = openmm.XmlSerializer.serialize(system)
# xml_file = open('cis_pro_xmlsystem.txt','w')
# xml_file.write(xml)
# xml_file.close()

temperature = 300.0 * unit.kelvin
collision_rate = 1.0 / unit.picosecond
timestep = 1.0 * unit.femtosecond
reportInterval = 1000
steps = 1E+8
fname = 'trans_pro_300K_noconstr_long'
#time = (steps*timestep).value_in_unit(unit.nanosecond)
parametersdict = {'Collision rate':collision_rate,'Temperature':temperature,'Timestep':timestep,'Report Interval':reportInterval}
import pickle
f_p = open(f'parameters/parameters{fname}.pkl','wb')
pickle.dump(parametersdict,f_p)
f_p.close

integrator = openmm.LangevinIntegrator(temperature,collision_rate,timestep)
#integrator.setConstraintTolerance(0.00001)
#integrator = openmm.VerletIntegrator(timestep)
properties_dict = {}
properties_dict["DeviceIndex"] = "2"
platform = openmm.Platform.getPlatform(2)

positions = pdb.getPositions(asNumpy=True).value_in_unit(unit.nanometer)

simulation = app.Simulation(pdb.getTopology(), system, integrator,platform,platformProperties=properties_dict)
#print(platform.getPropertyValue(simulation.context,property='Precision'))
#print(simulation.context.getPlatform())
simulation.context.setPositions(positions)
simulation.minimizeEnergy()
simulation.context.setVelocitiesToTemperature(temperature)

simulation.reporters.append(app.StateDataReporter(stdout, reportInterval, step=True, potentialEnergy=True,temperature=True))
simulation.reporters.append(openmm.app.StateDataReporter(
    f'Trajectories/{fname}.txt', 
    reportInterval=reportInterval, 
    step=True, 
    potentialEnergy=True,
    kineticEnergy = True,
    totalEnergy = True,
    temperature=True,
))
simulation.reporters.append(app.DCDReporter(f'Trajectories/{fname}.dcd',reportInterval))
#h5_reporter = reporters.HDF5Reporter('output.h5',reportInterval)
#simulation.reporters.append(h5_reporter)

simulation.step(steps)
#h5_reporter.close()



