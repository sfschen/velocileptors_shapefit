# velocileptors_shapefit
Taylor series emulator of velocileptors given shapefit parameters.

To make the emulator (mpi enabled) run

srun -n X -c Y python make_emulator_pells.py <basedir> <zeff> <Omfid>

The last two are given so that sigma8(z) is something sensible, i.e. for a given template with sigma8(0) we have
  
f(z) = fsigma8 / (sigma8(0) * D(z,OmegaM))
  
where fsigma8 is the input parameter. The taylor series is then stored in 
 
<basedir>/emu/shapefit_z_X.XX_Om_Y.YY_pkells.json.
  
In order to run the emulator one simply enters:
  
emu = Emulator_Pells('emu/shapefit_z_%.2f_Om_&.2f_pkells.json'%(z,Omfid),order=4)
kvec, p0, p2, p4 = emu(cpars, bpars)
  
here cpars are the cosmo parametrs (fs8, apar, aperp, m) and bpars are the bias terms in velocileptors.
  
To run this you will need to have \url{https://github.com/sfschen/velocileptors}. 
