#!/usr/bin/env python

import os
import json
import sys
from zlib import MAX_WBITS
import numpy as np
import logging
import argparse
from mtuq import read, open_db, download_greens_tensors
from mtuq.event import Origin, MomentTensor
from mtuq.graphics import plot_data_greens1, plot_beachball, plot_misfit_lune, plot_polarities, plot_misfit_dc
from mtuq.grid import FullMomentTensorGridSemiregular
from mtuq.grid_search import grid_search
from mtuq.misfit import WaveformMisfit,PolarityMisfit
from mtuq.process_data import ProcessData
from mtuq.util import fullpath, merge_dicts, save_json, sort_polarities
#from mtuq.util import sort_polarities
from mtuq.util.cap import parse_station_codes, Trapezoid
from mtuq.util.math import  to_v_w, to_rho, to_mij, from_mij
import glob

# Set up logging
logging.basicConfig(level=logging.INFO)
logging.getLogger('matplotlib.mathtext').setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

class Header:
    def __init__(self,station,component,time_shift,cc):
        self.station = station
        self.component = component
        self.time_shift = time_shift
        self.cc = cc

def _getattr(trace, name, *args):

    if len(args)==1:
        if not hasattr(trace, 'attrs'):
            return args[0]
        else:
            return getattr(trace.attrs, name, args[0])
    elif len(args)==0:
        return getattr(trace.attrs, name)
    else:
        raise TypeError("Wrong number of arguments")

def get_headerinfo(data,greens,misfit,stations,origin,source):

    synthetics = misfit.collect_synthetics(data, greens.select(origin), source)

    header_info = []

    for _i in range(len(stations)):
        stream_dat = data[_i]
        stream_syn = synthetics[_i] 

        for dat in stream_dat:
            component = dat.stats.channel[-1].upper()
            try:
                syn = stream_syn.select(component=component)[0]
            except:
                warn('Missing component, skipping...')
                continue

            time_shift = 0.
            time_shift += _getattr(syn, 'time_shift', np.nan)
            time_shift += _getattr(dat, 'static_time_shift', 0)

            s = syn.data
            d = dat.data
            # display maximum cross-correlation coefficient
            Ns = np.dot(s,s)**0.5
            Nd = np.dot(d,d)**0.5

            if Ns*Nd > 0.:
                max_cc = np.correlate(s, d, 'valid').max()
                max_cc /= (Ns*Nd)
            else:
                max_cc = np.nan
                
            header_info.append(Header(stations[_i]['station'],component,np.round(time_shift,2),np.round(max_cc,2)))
            print('{},{}: {} {}'.format(stations[_i]['station'],component,np.round(time_shift,2),np.round(max_cc,2)))

    return(header_info)

def wrap_up(ts_list,cc_list,station):
    total_ts = np.round(np.nansum(np.abs(ts_list))/(ts_list.size - np.count_nonzero(np.isnan(ts_list))),2)
    total_cc = np.round(np.nansum(cc_list)/(cc_list.size - np.count_nonzero(np.isnan(cc_list))),2)
            
    line = '{} {} {} {} {} {} {} {} {}'.format(station,ts_list[0],ts_list[1],ts_list[2],total_ts,cc_list[0],cc_list[1],cc_list[2],total_cc)
    return(line)

def write_headers(header_info,event_id):

    open_header_file=open('{}FMT_header_info.txt'.format(event_id),'w')

    open_header_file.write('STATIONS  ts_Z ts_R ts_T abs_av_shift cc_Z cc_R cc_T av_cc\n')

    init_stat = header_info[0].station
    ts_list = np.array([np.nan,np.nan,np.nan])
    cc_list = np.array([np.nan,np.nan,np.nan])

    for i in range(len(header_info)):

        if header_info[i].station != init_stat:

            line = wrap_up(ts_list,cc_list,init_stat)
            open_header_file.write(line+'\n')
            print(line)

            ts_list = np.array([np.nan,np.nan,np.nan])
            cc_list = np.array([np.nan,np.nan,np.nan])

            init_stat = header_info[i].station

        if header_info[i].station == init_stat:
            if header_info[i].component == 'Z':
                ts_list[0] = header_info[i].time_shift
                cc_list[0] = header_info[i].cc

            if header_info[i].component == 'R':
                ts_list[1] = header_info[i].time_shift
                cc_list[1] = header_info[i].cc

            if header_info[i].component == 'T':
                ts_list[2] = header_info[i].time_shift
                cc_list[2] = header_info[i].cc

            if i == len(header_info)-1:
                line = wrap_up(ts_list,cc_list,header_info[i].station)
                open_header_file.write(line)
                print(line)

def mw2mo(mw):
    #inverse of https://github.com/mtuqorg/mtuq/blob/0a09059cfb2dbf91a4ab9b41ff8b820b786e9d78/mtuq/event.py#L144
    exponent = (1.5*mw + 9.1)
    mo=np.power(10, exponent)
    return mo

if __name__=='__main__':
    
    # Read the moment tensor Json file
    config = f'input_source.json'
    #Extract the values v,w,kappa,sigma,h from the input_source.json file
    with open(config, 'r') as f:
        src= json.load(f)

    
    # 1. Calculate Scalar Moment (M0) from Mw
    m0 = mw2mo(src['mw'])

    kappa = src['kappa']
    sigma = src['sigma']
    h = src['h']

    # 2. Calculate Tensor components (Mij) 
    if src['format'] == 'vw':
        print('Moment Tensor format: v-w')

        # Define v and w so they are available for lune_dict later
        v = src['v']
        w = src['w']

        mij = to_mij(
            rho=m0*np.sqrt(2.),  # Tape2012,
            v=v, 
            w=w, 
            kappa=src['kappa'], 
            sigma=src['sigma'], 
            h=src['h']
        )
        print(mij)

    elif src['format'] == 'gamma_delta':
        v, w = to_v_w(src['delta'],src['gamma'])
        mij = to_mij(
            rho=m0*np.sqrt(2.),
            v=v,
            w=w,
            kappa=src['kappa'],
            sigma=src['sigma'],
            h=src['h']
        )

    elif src['format'] == 'USE':
        mt_use = np.array([src['Mrr'], src['Mtt'], src['Mpp'], src['Mrt'], src['Mrp'], src['Mtp']])
        mo_use = (1 / np.sqrt(2)) * np.sqrt(mt_use[0]**2 + mt_use[1]**2 + mt_use[2]**2 + 2 * (mt_use[3]**2 + mt_use[4]**2 + mt_use[5]**2))
        mt_use_normalized = mt_use / mo_use
        rho_test, v, w, kappa, sigma, h = from_mij(mt_use_normalized)
        rho=m0*np.sqrt(2.)
        mij = to_mij(rho, v, w, kappa, sigma, h)

    elif src['format'] == 'NED':
        mt_ned = np.array([src['Mzz'], src['Mxx'], src['Myy'], src['Mxz'], -1*src['Myz'], -1*src['Mxy']])
        mo_ned = (1 / np.sqrt(2)) * np.sqrt(mt_ned[0]**2 + mt_ned[1]**2 + mt_ned[2]**2 + 2 * (mt_ned[3]**2 + mt_ned[4]**2 + mt_ned[5]**2))
        mt_ned_normalized = mt_ned / mo_ned
        rho_test, v, w, kappa, sigma, h = from_mij(mt_ned_normalized)
        rho=m0*np.sqrt(2.)
        mij = to_mij(rho, v, w, kappa, sigma, h)

    else:
        logger.error(f"Source format {src['format']} not recognized.")
        logger.info(f"Available formats: 'vw', 'gamma_delta', 'USE', 'NED'.")
        sys.exit(1)
    
    
    time = src['time']
    evla = src['latitude']
    evlo = src['longitude']
    evdp = src['depth_in_m']
    event = src['id']
    magnitude = src['mw']
    rho = to_rho(magnitude)

    mt = MomentTensor(mij,convention="USE")
    
    lune_dict = {'rho': rho, 'v': v, 'w': w, 
                 'kappa': kappa, 'sigma': sigma, 'h': h}


    #define origin and magnitude
    origin = Origin({
            'time': f'{time}',
            'latitude': f'{evla}',
            'longitude': f'{evlo}',
            'depth_in_m': f'{evdp}',
            })
    
    wavelet = Trapezoid(
        magnitude=np.average(magnitude))

    #Read data parameters
    mdir = os.getcwd()
    path_data=    fullpath('{}/{}/*.[zrt]'.format(mdir,event))
    path_weights= fullpath('{}/{}/weights.dat'.format(mdir,event))
    event_id=     '{}'.format(event)
    station_id_list = parse_station_codes(path_weights)

    mdir = os.getcwd()

    #MAKE THE DIRECTORY STRUCTURE
    #FORWARD_SOLUTIONS/FK_GF/DATA_TESTS/20250620174913_Mw_4.9_D_7km_F_15-33_200/MT_1
    depth = float(evdp)
    T_min = float(src['fb'].split('-')[0])
    T_max = float(src['fb'].split('-')[1]) 
    wlen = float(src['wl'])

    GF = src["model"].split("-")[0]
    vel_model = src["model"].split("-")[1]

    solution_dir = f'{mdir}/RESULTS/{GF}_GF/{vel_model}/DATA_TESTS/{event}_Mw_{magnitude}_D_{depth/1000}km_F_{T_min}-{T_max}_wl_{wlen}/'
    # Make a list with the number of MT solutions inside solution_dir
    list_MT = glob.glob(f'{solution_dir}MT_*')
    if len(list_MT) == 0:
        solution_dir += 'MT_1'
    else:
        nums = [int(num.split('_')[-1]) for num in list_MT]
        print('There are {} MT solutions in {}'.format(len(nums), solution_dir))
        last_number = max(nums)
        solution_dir += f'MT_{last_number+1}'   


    path_data=    fullpath('{}/{}/*.[zrt]'.format(mdir,event))
    path_weights= fullpath('{}/{}/weights.dat'.format(mdir,event))
    event_id=     '{}'.format(event)
    model=  'ak135'
    
    if GF == 'FK':
        db = open_db('{}/{}/{}'.format(mdir,GF,vel_model),format='FK')
    elif GF == 'SPECFEM3D':
        db = open_db('{}/{}/'.format(vel_model,event_id),format="SPECFEM3D")
    else:
        logger.error(f"Green's function format {GF} not recognized.")
        logger.info(f"Available formats: 'FK', 'SPECFEM3D'.")
        sys.exit(1)
 

    #
    # Body and surface wave measurements will be made separately
    #
    
    #Body waves are only used for using the subroutine of predicting the polarities

    if GF == 'FK':

        process_bw = ProcessData(
            filter_type='Bandpass',
            freq_min= 1/float(10),
            freq_max= 1/float(3),
            pick_type='FK_metadata',
            FK_database='{}/{}/{}'.format(mdir,GF,vel_model),
            window_type='body_wave',
            window_length=15,
            capuaf_file=path_weights,
            )
        
        process_sw = ProcessData(
            filter_type='Bandpass',
            freq_min=1/T_max,
            freq_max=1/T_min,
            pick_type='FK_metadata',
            FK_database='{}/{}/{}'.format(mdir,GF,vel_model),
            window_type='surface_wave',
            window_length=wlen,
            capuaf_file=path_weights,
            )
        
    elif GF == 'SPECFEM3D':

        process_bw = ProcessData(
            filter_type='Bandpass',
            freq_min= 1/float(10),
            freq_max= 1/float(3),
            pick_type='taup',
            taup_model=model,
            window_type='body_wave',
            window_length=15,
            capuaf_file=path_weights,
            )
        
        process_sw = ProcessData(
            filter_type='Bandpass',
            freq_min=1/T_max,
            freq_max=1/T_min,
            pick_type='taup',
            taup_model=model,
            window_type='surface_wave',
            window_length=wlen,
            capuaf_file=path_weights,
            ) 

    
    misfit_sw = WaveformMisfit(
        norm='L2',
        time_shift_min=-15.,
        time_shift_max=+15.,
        time_shift_groups=['ZR','T'],
        )
    
    polarity_misfit = PolarityMisfit(
        taup_model=model)


    #
    # User-supplied weights control how much each station contributes to the
    # objective function
    #
    
    station_id_list = parse_station_codes(path_weights)

    #Read data
    print('Reading data...\n')

    data = read(path_data, format='sac', 
                event_id=event_id,
                station_id_list=station_id_list,
                tags=['units:cm', 'type:velocity'])
    
    data.sort_by_distance()
    stations = data.get_stations()

    #Process data
    print('Processing data...\n')
    data_sw = data.map(process_sw)

    print('Reading Greens functions...\n')
    greens = db.get_greens_tensors(stations,origin)

    print('Processing Greens functions...\n')
    greens.convolve(wavelet)
    greens_bw = greens.map(process_bw)
    greens_sw = greens.map(process_sw)


    #Dealing with polarities

    #First, a zeroes polarity numpy array is created with the same length of the stations in data.
    #number_stations = len(data)
    #polarities= np.zeros(number_stations)

    #Second, read the polarities.json file inside the event directory.
    #file_path = '{}/polarities.json'.format(event)
    #with open(file_path, 'r') as file:
    #    dict_polarity  = json.load(file)
   
    #Third: the polarities array is populated with the polarities in the dictionary
    #the subroutine sort_polarities helps to ensure that the order of the entered polarities
    #is the same than the order of stations in data.

    #polarities = sort_polarities(dict_polarity,data,polarities)

    print('Generating figures...\n')

    plot_data_greens1(event_id+'_Forward_waveforms.pdf',
            data_sw, greens_sw, process_sw,
            misfit_sw, stations, origin, mt, lune_dict)

    #header_info = get_headerinfo(data_sw,greens_sw,misfit_sw,stations,origin,mt)

    #write_headers(header_info,event_id)

    plot_beachball(event_id+'_Forward_beachball.pdf',
            mt, stations, origin, taup_model=model)

    # generate polarity figures

    # predicted polarities
    #predicted = polarity_misfit.get_predicted(greens, mt)

    # station attributes
    #attrs = polarity_misfit.collect_attributes(polarities, greens)

    #plot_polarities(event_id+'_Forward_beachball_polarity.pdf',
    #        polarities, predicted, attrs, origin, mt, taup_model=model)

    print('Saving results...\n')

    print('Saving files in directory: {}'.format(solution_dir))
    if not os.path.exists(solution_dir):
        os.makedirs(solution_dir)

    cp_solution1=print('mv {}* {}'.format(event_id+'_Forward_',solution_dir))
    os.system('mv {}* {}'.format(event_id+'_Forward_',solution_dir))

    #cp_polarities = print('cp {}/polarities.json {}'.format(event_id,solution_dir))
    #os.system('cp {}/polarities.json {}'.format(event_id,solution_dir))

    cp_json = print('cp input_source.json {}'.format(solution_dir))
    os.system('cp input_source.json {}'.format(solution_dir))

    #cp_headerinfo = print('mv {}FMT_header_info.txt {}'.format(event_id,solution_dir))
    #os.system('mv {}FMT_header_info.txt {}'.format(event_id,solution_dir))  

    print('\nFinished\n')
