#!/usr/bin/env python

import os
import numpy as np
from mpi4py import MPI

def create_FK_greens():
    '''Create Greens' by using FK in parallel via MPI.'''

    # Initialize MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()  # The ID of the current processor (e.g., 0, 1, 2...)
    size = comm.Get_size()  # Total number of processors being used

    # Define the full search space
    searching_depths = np.arange(1, 31, 1)       
    searching_distances = np.arange(1, 501, 1)

    # Split the depths evenly across the available processors
    # Each processor will only loop through its assigned 'local_depths'
    local_depths = np.array_split(searching_depths, size)[rank]

    # set model parameters.
    fk_command    = 'fk.pl'
    model_name    = 'wus'
    model_type    = 'f'
    npts          = 4096         
    dt            = 0.1
    src_type      = ['0', '2']  
    is_sr_dist_degree = False

    # ONLY Rank 0 should create the main directory to avoid conflicts
    if rank == 0:
        mkdir_gf_dir = 'mkdir -p {}_GF'.format(model_name)
        print(f"Rank 0: {mkdir_gf_dir}")
        os.system(mkdir_gf_dir)

    # Tell all other processors to wait here until Rank 0 finishes making the folder
    comm.Barrier()

    # Create the Greens functions for this processor's specific depths
    for d in local_depths:
        for s_type in src_type:
            cmd_str = "%s -M%s/%d/%s -N%d/%.4f -S%s " % (fk_command, model_name, d, model_type, npts, dt, s_type)
            
            if is_sr_dist_degree:
                cmd_str += '-D '
            
            for sr_d in searching_distances:
                cmd_str += str(" %d " % sr_d)

            # Print which rank is running what, to keep track of progress
            print(f"Rank {rank} calculating depth {d} km, Source {s_type}")
            os.system(cmd_str)
            
        # Move the completed depth folder into the master directory
        mv_dir = 'mv {}_{} {}_GF/'.format(model_name, int(d), model_name)
        os.system(mv_dir)

    # Make sure all processors finish before the script exits
    comm.Barrier()
    if rank == 0:
        print("\nAll Green's Functions calculated successfully in parallel!")

if __name__=='__main__':
    create_FK_greens()