import numpy as np
import json
import sys
import os
from mpi4py import MPI

from linear_theory import f_of_a, D_of_a
from taylor_approximation import compute_derivatives

comm = MPI.COMM_WORLD
mpi_rank = comm.Get_rank()
mpi_size = comm.Get_size()
if mpi_rank==0:
    print(sys.argv[0]+" running on {:d} processes.".format(mpi_size))
#print( "Hello I am process %d of %d." %(mpi_rank, mpi_size) )

basedir = sys.argv[1] +'/'
z = float(sys.argv[2])
Omfid = float(sys.argv[3])

# Set up the output k vector:
from compute_pell_tables_template import compute_pell_tables, kvec, sigma8_z0

output_shape = (len(kvec),19) # two multipoles and 19 types of terms

# First construct the grid

order = 4
# these are OmegaM, h, sigma8
fs80 = f_of_a(1./(1+z),OmegaM=Omfid) * D_of_a(1./(1+z),OmegaM=Omfid) * sigma8_z0
print("Expanding about fsigma8 = %.2f" %(fs80))
x0s = [fs80, 1.0, 1.0, 0]; Nparams = len(x0s) # these are chosen to be roughly at the BOSS best fit value
dxs = [0.05, 0.01, 0.01, 0.01]

template = np.arange(-order,order+1,1)
Npoints = 2*order + 1
grid_axes = [ dx*template + x0 for x0, dx in zip(x0s,dxs)]

Inds   = np.meshgrid( * (np.arange(Npoints),)*Nparams, indexing='ij')
Inds = [ind.flatten() for ind in Inds]
center_ii = (order,)*Nparams
Coords = np.meshgrid( *grid_axes, indexing='ij')

P0grid = np.zeros( (Npoints,)*Nparams+ output_shape)
P2grid = np.zeros( (Npoints,)*Nparams+ output_shape)
P4grid = np.zeros( (Npoints,)*Nparams+ output_shape)

P0gridii = np.zeros( (Npoints,)*Nparams+ output_shape)
P2gridii = np.zeros( (Npoints,)*Nparams+ output_shape)
P4gridii = np.zeros( (Npoints,)*Nparams+ output_shape)

for nn, iis in enumerate(zip(*Inds)):
    if nn%mpi_size == mpi_rank:
        coord = [Coords[i][iis] for i in range(Nparams)]
        print(coord,iis)
        p0, p2, p4 = compute_pell_tables(coord,zeff=z,Omfid=Omfid)
        
        P0gridii[iis] = p0
        P2gridii[iis] = p2
        P4gridii[iis] = p4
        
comm.Allreduce(P0gridii, P0grid, op=MPI.SUM)
comm.Allreduce(P2gridii, P2grid, op=MPI.SUM)
comm.Allreduce(P4gridii, P4grid, op=MPI.SUM)

del(P0gridii, P2gridii, P4gridii)

# Now compute the derivatives
derivs0, derivs2, derivs4 = 0, 0, 0

if mpi_rank == 1:
    print("Computing derivs0.")
    derivs0 = compute_derivatives(P0grid, dxs, center_ii, 4)
if mpi_rank == 2:
    print("Computing derivs2.")
    derivs2 = compute_derivatives(P2grid, dxs, center_ii, 4)
if mpi_rank == 3:
    print("Computing derivs4.")
    derivs4 = compute_derivatives(P4grid, dxs, center_ii, 4)

comm.Barrier()

derivs0 = comm.bcast(derivs0, root=1)    
derivs2 = comm.bcast(derivs2, root=2)
derivs4 = comm.bcast(derivs4, root=3)

print("Derivatives computed.")

# Save

if mpi_rank == 0:
    # Make the emulator (emu) directory if it
    # doesn't already exist.
    fb = basedir+'emu'
    if not os.path.isdir(fb):
        print("Making directory ",fb)
        os.mkdir(fb)
    else:
        print("Found directory ",fb)
    #
comm.Barrier()

# Now save:
outfile = basedir + 'emu/shapefit_z_%.2f_Om_%.2f_pkells.json'%(z,Omfid)

list0 = [ dd.tolist() for dd in derivs0 ]
list2 = [ dd.tolist() for dd in derivs2 ]
list4 = [ dd.tolist() for dd in derivs4 ]

outdict = {'params': ['f','apar','aperp','m'],\
           'x0': x0s,\
           'kvec': kvec.tolist(),\
           'derivs0': list0,\
           'derivs2': list2,\
           'derivs4': list4}

if mpi_rank == 0:
    json_file = open(outfile, 'w')
    json.dump(outdict, json_file)
    json_file.close()

