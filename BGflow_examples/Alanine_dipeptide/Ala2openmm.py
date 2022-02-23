
from simtk import unit, openmm
from simtk.openmm import app
import numpy as np
from sys import stdout
from openmmtools import integrators


pdb = app.PDBFile('ala2_fromURL.pdb')
topology = pdb.getTopology()
positions = pdb.getPositions()


ff = app.ForceField('amber99sbildn.xml',"amber96_obc.xml")
system = ff.createSystem(
    topology=topology, 
    constraints=app.HBonds, 
    rigidWater=True
    )

platform = openmm.Platform.getPlatform(2)

# Uncomment this cell to save the system as an xml file
# xml = openmm.XmlSerializer.serialize(system)
# xml_file = open('ala2_xml_system.txt','w')
# xml_file.write(xml)
# xml_file.close()

temperature = 1000.0 * unit.kelvin
collision_rate = 1.0 / unit.picosecond
timestep = 1.0 * unit.femtosecond
reportInterval = 2500
steps = 5E+8
fname = '1000K'
#time = (steps*timestep).value_in_unit(unit.nanosecond)
parametersdict = {'Collision rate':collision_rate,'Temperature':temperature,'Timestep':timestep,'Report Interval':reportInterval}
import pickle
f_p = open(f'parameters{fname}.pkl','wb')
pickle.dump(parametersdict,f_p)
f_p.close

integrator = integrators.LangevinIntegrator(temperature=temperature,collision_rate=collision_rate,timestep=timestep)
#integrator.setConstraintTolerance(0.00001)
#integrator = openmm.VerletIntegrator(timestep)
properties_dict = {}
properties_dict["DeviceIndex"] = "2"
simulation = app.Simulation(topology, system, integrator,platform,platformProperties=properties_dict)
#print(platform.getPropertyValue(simulation.context,property='Precision'))
#print(simulation.context.getPlatform())
simulation.context.setPositions(positions)
simulation.minimizeEnergy()
simulation.context.setVelocitiesToTemperature(temperature)

simulation.reporters.append(app.StateDataReporter(stdout, reportInterval=reportInterval*10, step=True, potentialEnergy=True,temperature=True))
simulation.reporters.append(app.DCDReporter(f'{fname}.dcd',reportInterval))
simulation.reporters.append(openmm.app.StateDataReporter(
    f'{fname}.txt', 
    reportInterval=reportInterval, 
    step=True, 
    potentialEnergy=True,
    kineticEnergy = True,
    totalEnergy = True,
    temperature=True,
))

simulation.step(steps)

