#!/usr/bin/env python

import os
import numpy as np
from mtuq import read
from mtuq.util import fullpath
from mtuq.util.cap import parse_station_codes
import argparse


def create_FK_greens():
    
    '''Create Greens' by using FK. '''

    searching_depths = np.arange(1,6,1)       # eg: from 8 to 13 km with interval of 1 km
    searching_distances = np.arange(1,6,1) 
    # set model parameters.
    fk_command    = 'fk.pl'
    model_name    = 'wus'
    model_type    = 'f'
    npts          = 4096         # must be 2^n
    dt            = 0.1
    src_type      = ['0', '2']  # 0-Explosion source, 2-Double-couple source
    is_sr_dist_degree = False

    mkdir_gf_dir = 'mkdir -p {}_GF'.format(model_name)
    print(mkdir_gf_dir)
    os.system(mkdir_gf_dir)

    # create the Greens function
    for d in searching_depths:
        for s_type in src_type:
            cmd_str = "%s -M%s/%d/%s -N%d/%.4f -S%s " % (fk_command, model_name, d, model_type, npts, dt, s_type)
            # if source-receiver distance is degree, otherwise is km.
            if is_sr_dist_degree:
                cmd_str += '-D '
            # add source-receiver distance
            for sr_d in searching_distances:
                cmd_str += str(" %d " % sr_d)

            # create Green's function by using FK.
            print(cmd_str)
            os.system(cmd_str)
            
        mv_dir = 'mv {}_{} {}_GF/'.format(model_name,int(d),model_name)
        os.system(mv_dir)

if __name__=='__main__':
    create_FK_greens()
