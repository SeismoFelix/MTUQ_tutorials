#!/usr/bin/env python

import os
import sys
import json
import numpy as np

from mtuq import read, open_db, download_greens_tensors
from mtuq.event import Origin
from mtuq.graphics import (plot_data_greens1, plot_data_greens2, plot_misfit_depth,
                           plot_misfit_dc, plot_beachball, plot_misfit_lune,
                           plot_variance_reduction_lune, plot_variance_reduction_dc,
                           plot_polarities)
from mtuq.grid import DoubleCoupleGridRegular, DeviatoricGridSemiregular, FullMomentTensorGridSemiregular
from mtuq.grid_search import grid_search
from mtuq.misfit import WaveformMisfit, PolarityMisfit
from mtuq.process_data import ProcessData
from mtuq.util import fullpath, merge_dicts, save_json, sort_polarities
from mtuq.util.cap import parse_station_codes, Trapezoid
from mtuq.misfit.waveform import calculate_norm_data
from mpi4py import MPI

class Header:
    def __init__(self, station, component, time_shift, cc):
        self.station = station
        self.component = component
        self.time_shift = time_shift
        self.cc = cc

def _getattr(trace, name, *args):
    if len(args) == 1:
        if not hasattr(trace, 'attrs'):
            return args[0]
        else:
            return getattr(trace.attrs, name, args[0])
    elif len(args) == 0:
        return getattr(trace.attrs, name)
    else:
        raise TypeError("Wrong number of arguments")

def get_headerinfo(data, greens, misfit, stations, origin, source):
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
                print('Missing component, skipping...')
                continue

            time_shift = 0.
            time_shift += _getattr(syn, 'time_shift', np.nan)
            time_shift += _getattr(dat, 'static_time_shift', 0)

            s = syn.data
            d = dat.data
            
            Ns = np.dot(s, s)**0.5
            Nd = np.dot(d, d)**0.5

            if Ns * Nd > 0.:
                max_cc = np.correlate(s, d, 'valid').max()
                max_cc /= (Ns * Nd)
            else:
                max_cc = np.nan

            header_info.append(Header(stations[_i]['station'], component, np.round(time_shift, 2), np.round(max_cc, 2)))
            print(f"{stations[_i]['station']},{component}: {np.round(time_shift, 2)} {np.round(max_cc, 2)}")

    return header_info

def wrap_up(ts_list, cc_list, station):
    valid_ts_count = ts_list.size - np.count_nonzero(np.isnan(ts_list))
    total_ts = np.round(np.nansum(np.abs(ts_list)) / valid_ts_count, 2) if valid_ts_count > 0 else np.nan
    
    valid_cc_count = cc_list.size - np.count_nonzero(np.isnan(cc_list))
    total_cc = np.round(np.nansum(cc_list) / valid_cc_count, 2) if valid_cc_count > 0 else np.nan

    line = f"{station} {ts_list[0]} {ts_list[1]} {ts_list[2]} {total_ts} {cc_list[0]} {cc_list[1]} {cc_list[2]} {total_cc}"
    return line

def write_headers(header_info, event_id, tag, grid_type):
    filename = f'{event_id}{grid_type}_header_info_{tag}.txt'
    with open(filename, 'w') as open_header_file:
        open_header_file.write('STATIONS  ts_Z ts_R ts_T abs_av_shift cc_Z cc_R cc_T av_cc\n')

        init_stat = header_info[0].station
        ts_list = np.array([np.nan, np.nan, np.nan])
        cc_list = np.array([np.nan, np.nan, np.nan])

        for i in range(len(header_info)):
            if header_info[i].station != init_stat:
                line = wrap_up(ts_list, cc_list, init_stat)
                open_header_file.write(line + '\n')
                ts_list = np.array([np.nan, np.nan, np.nan])
                cc_list = np.array([np.nan, np.nan, np.nan])
                init_stat = header_info[i].station

            if header_info[i].station == init_stat:
                comp_map = {'Z': 0, 'R': 1, 'T': 2}
                if header_info[i].component in comp_map:
                    idx = comp_map[header_info[i].component]
                    ts_list[idx] = header_info[i].time_shift
                    cc_list[idx] = header_info[i].cc

                if i == len(header_info) - 1:
                    line = wrap_up(ts_list, cc_list, header_info[i].station)
                    open_header_file.write(line)


if __name__ == '__main__':
    # Initialize MPI
    comm = MPI.COMM_WORLD

    if len(sys.argv) > 1:
        config_file = sys.argv[1]
    else:
        config_file = 'parameters_inversion.json'

    with open(config_file, 'r') as f:
        params = json.load(f)

    event_id = params['id']
    mdir = os.getcwd()

    path_data = f'{mdir}/{event_id}/*.[zrt]'
    path_weights = f'{mdir}/{event_id}/weights.dat'
    
    # Grid parameters
    grid_type = params.get('grid_type', 'DEV').upper()
    ppa = int(params['npts_per_axis'])
    use_polarities = params.get('use_polarities', False)
    
    # Determine DB Engine and Model paths
    engine, submodel = params.get('model', 'FK-ic').split('-')
    taup_model = params['model_taup']

    if engine == 'FK':
        db = open_db(f'{mdir}/FK/{submodel}', format='FK')
        pick_type_cfg = 'FK_metadata'
        FK_db_cfg = f'{mdir}/FK/{submodel}'
        taup_cfg = None
    elif engine == 'SPECFEM3D':
        db = open_db(f'{mdir}/3D_{submodel}/{event_id}', format='SPECFEM3D')
        pick_type_cfg = 'taup'
        FK_db_cfg = None
        taup_cfg = taup_model
    elif engine == 'SYNGINE':
        db = None
        pick_type_cfg = 'taup'
        FK_db_cfg = None
        taup_cfg = taup_model
    else:
        if comm.rank == 0: 
            print(f"Unrecognized engine: {engine}")
        sys.exit(1)

    # ProcessData Setup
    sw_cfg = params.get('surface_waves', {'use': False})
    bw_cfg = params.get('body_waves', {'use': False})

    use_sw = sw_cfg.get('use', False)
    use_bw = bw_cfg.get('use', False)

    if use_sw:
        process_sw = ProcessData(
            filter_type='Bandpass',
            freq_min=1.0 / float(sw_cfg['period_max']),
            freq_max=1.0 / float(sw_cfg['period_min']),
            pick_type=pick_type_cfg,
            FK_database=FK_db_cfg,
            taup_model=taup_cfg,
            window_type='surface_wave',
            window_length=float(sw_cfg['window_length']),
            capuaf_file=path_weights,
        )
        misfit_sw = WaveformMisfit(
            norm='L2',
            time_shift_min=float(sw_cfg['time_shift_min']),
            time_shift_max=float(sw_cfg['time_shift_max']),
            time_shift_groups=['ZR', 'T'],
        )

    if use_bw or use_polarities:
        # Fallbacks if polarities are used but BW isn't strictly requested for waveform misfit
        bw_p_max = float(bw_cfg.get('period_max', 15.0)) if use_bw else 10.0
        bw_p_min = float(bw_cfg.get('period_min', 3.0)) if use_bw else 3.0
        bw_wlen = float(bw_cfg.get('window_length', 30.0)) if use_bw else 15.0

        process_bw = ProcessData(
            filter_type='Bandpass',
            freq_min=1.0 / bw_p_max,
            freq_max=1.0 / bw_p_min,
            pick_type=pick_type_cfg,
            FK_database=FK_db_cfg,
            taup_model=taup_cfg,
            window_type='body_wave',
            window_length=bw_wlen,
            capuaf_file=path_weights,
        )
        if use_bw:
            misfit_bw = WaveformMisfit(
                norm='L2',
                time_shift_min=float(bw_cfg['time_shift_min']),
                time_shift_max=float(bw_cfg['time_shift_max']),
                time_shift_groups=['ZR'],
            )

    if use_polarities:
        polarity_misfit = PolarityMisfit(taup_model=taup_model)

    station_id_list = parse_station_codes(path_weights)

    # Grid Construction
    mw_init = float(params['mw_init'])
    mw_fin = float(params['mw_fin'])
    mw_step = float(params['mw_step'])
    magnitudes = np.arange(mw_init, mw_fin + mw_step/10.0, mw_step)

    depth_min = float(params['depth_min'])
    depth_max = float(params['depth_max'])
    depth_step = float(params['depth_step'])
    prem_depths = np.arange(depth_min * 1000, (depth_max * 1000) + (depth_step * 1000)/10.0, depth_step * 1000)
    
    depths = np.array([d for d in prem_depths if 1000 <= d <= 49000])

    catalog_origin = Origin({
        'time': params['origin_time'],
        'latitude': float(params['latitude']),
        'longitude': float(params['longitude']),
        'depth_in_m': np.average(depths) if len(depths) > 0 else 0,
    })

    origins = [catalog_origin.copy() for d in depths]
    for i, origin in enumerate(origins):
        setattr(origin, 'depth_in_m', depths[i])

    if grid_type == 'DC':
        grid = DoubleCoupleGridRegular(npts_per_axis=ppa, magnitudes=magnitudes)
    elif grid_type == 'DEV':
        grid = DeviatoricGridSemiregular(npts_per_axis=ppa, magnitudes=magnitudes)
    elif grid_type == 'FMT':
        grid = FullMomentTensorGridSemiregular(npts_per_axis=ppa, magnitudes=magnitudes)

    wavelet = Trapezoid(magnitude=np.average(magnitudes))

    #
    # Data I/O
    #
    if comm.rank == 0:
        print('Reading data...\n')
        data = read(path_data, format='sac',
                    event_id=event_id,
                    station_id_list=station_id_list,
                    tags=[f"units:{params.get('units', 'cm')}", f"type:{params.get('type', 'velocity')}"])
        data.sort_by_distance()
        stations = data.get_stations()

        if engine == 'SYNGINE':
            print('Downloading Greens functions from SYNGINE...\n')
            greens = download_greens_tensors(stations, origins, submodel)
        else:
            print('Reading Greens functions...\n')
            greens = db.get_greens_tensors(stations, origins)
        
        greens.convolve(wavelet)

        if use_sw:
            data_sw = data.map(process_sw)
            greens_sw = greens.map(process_sw)
        if use_bw or use_polarities:
            data_bw = data.map(process_bw)
            greens_bw = greens.map(process_bw)
    else:
        stations = data_sw = greens_sw = data_bw = greens_bw = None

    stations = comm.bcast(stations, root=0)
    if use_sw:
        data_sw = comm.bcast(data_sw, root=0)
        greens_sw = comm.bcast(greens_sw, root=0)
    if use_bw or use_polarities:
        data_bw = comm.bcast(data_bw, root=0)
        greens_bw = comm.bcast(greens_bw, root=0)

    #
    # Misfit Evaluation
    #
    
    # Initialize the results container safely, ONLY on the master processor
    if comm.rank == 0:
        results = None

    if use_sw:
        if comm.rank == 0: 
            print('Evaluating surface wave misfit...\n')
            
        results_sw = grid_search(data_sw, greens_sw, misfit_sw, origins, grid)
        
        # Rank 0 securely stores the surface wave DataArray
        if comm.rank == 0:
            results = results_sw

    if use_bw:
        if comm.rank == 0: 
            print('Evaluating body wave misfit...\n')
            
        results_bw = grid_search(data_bw, greens_bw, misfit_bw, origins, grid)
        
        if comm.rank == 0:
            # If surface waves were also used, add the DataArrays together
            if results is not None:
                results += results_bw
            # If ONLY body waves were used, establish the baseline
            else:
                results = results_bw

    if use_polarities:
        if comm.rank == 0: 
            print('Evaluating polarity misfit...\n')
            
        number_stations = len(stations)
        polarities = np.zeros(number_stations)
        file_path = f'{mdir}/{event_id}/polarities.json'
        
        # Load local data dummy to sort polarities matching array
        data_pol = read(path_data, format='sac', event_id=event_id, station_id_list=station_id_list, tags=[f"units:{params.get('units', 'cm')}", f"type:{params.get('type', 'velocity')}"])
        data_pol.sort_by_distance()
        
        with open(file_path, 'r') as file:
            dict_polarity = json.load(file)
            
        polarities = sort_polarities(dict_polarity, data_pol, polarities)
        results_polarity = grid_search(polarities, greens_bw, polarity_misfit, origins, grid)

    #
    # Post-processing and Plotting
    #
    if comm.rank == 0:
        idx = results.source_idxmin()
        best_source = grid.get(idx)
        best_origin = origins[results.origin_idxmin()]
        lune_dict = grid.get_dict(idx)
        mt_dict = best_source.as_dict()

        merged_dict = merge_dicts(
            mt_dict, lune_dict, {'M0': best_source.moment(), 'Mw': best_source.magnitude()}, best_origin)

        print('Generating figures...\n')

        # Conditional Waveform Plotting
        if use_sw and use_bw:
            plot_data_greens2(f"{event_id}{grid_type}_waveforms.pdf",
                data_bw, data_sw, greens_bw, greens_sw, process_bw, process_sw, 
                misfit_bw, misfit_sw, stations, best_origin, best_source, lune_dict)
        elif use_sw:
            plot_data_greens1(f"{event_id}{grid_type}_waveforms.pdf",
                data_sw, greens_sw, process_sw, misfit_sw, stations, best_origin, best_source, lune_dict)

        # Depth Trade-offs
        if len(depths) > 1:
            plot_misfit_depth(f"{event_id}{grid_type}_misfit_depth.pdf", results, origins, title=event_id)
            plot_misfit_depth(f"{event_id}{grid_type}_misfit_depth_tradeoffs.pdf", results, origins, show_tradeoffs=True, show_magnitudes=True, title=event_id)

        plot_beachball(f"{event_id}{grid_type}_beachball.pdf", best_source, stations, best_origin, taup_model=taup_model)

        # Calculate Norm for Variance Reduction (Relies on SW if available)
        norm_sw = None
        if use_sw:
            norm_sw = calculate_norm_data(data_sw, misfit_sw.norm, ['Z', 'R', 'T'])

        # Universal Double-Couple Diagnostics (Applies to DC, DEV, and FMT)
        plot_misfit_dc(f"{event_id}{grid_type}_misfit_dc.pdf", results)
        if use_sw:
            plot_variance_reduction_dc(f"{event_id}{grid_type}_DC_variance_reduction.pdf", results, norm_sw, colorbar_label='VR (%)')

        # Lune Diagnostics (Applies strictly to DEV and FMT)
        if grid_type in ['DEV', 'FMT']:
            plot_misfit_lune(f"{event_id}{grid_type}_misfit.pdf", results)
            plot_misfit_lune(f"{event_id}{grid_type}_misfit_mt.pdf", results, show_mt=True)
            plot_misfit_lune(f"{event_id}{grid_type}_misfit_tradeoff.pdf", results, show_tradeoffs=True)
            if use_sw:
                plot_variance_reduction_lune(f"{event_id}{grid_type}_variance_reduction.pdf", results, norm_sw, colorbar_label='VR (%)', show_mt=True)

        # Polarity Diagnostics
        if use_polarities:
            plot_misfit_lune(f"{event_id}{grid_type}_misfit_polarity.pdf", results_polarity, show_best=False, title='Polarity Misfit', plot_type='scatter')
            predicted = polarity_misfit.get_predicted(greens_bw, best_source)
            attrs = polarity_misfit.collect_attributes(polarities, greens_bw)
            plot_polarities(f"{event_id}{grid_type}_beachball_polarity.pdf", polarities, predicted, attrs, best_origin, best_source, taup_model=taup_model)

        print('Saving results...\n')
        
        # Headers & Raw Files
        if use_sw:
            sw_header = get_headerinfo(data_sw, greens_sw, misfit_sw, stations, best_origin, best_source)
            write_headers(sw_header, event_id, 'sw', grid_type)
        if use_bw:
            bw_header = get_headerinfo(data_bw, greens_bw, misfit_bw, stations, best_origin, best_source)
            write_headers(bw_header, event_id, 'bw', grid_type)

        save_json(f"{event_id}{grid_type}_solution.json", merged_dict)
        origins_dict = {_i: origin for _i, origin in enumerate(origins)}
        save_json(f"{event_id}{grid_type}_origins.json", origins_dict)
        results.save(f"{event_id}{grid_type}_misfit.nc")

        # Organize Output Directory Dynamically
        period_str_sw = f"{sw_cfg['period_min']}-{sw_cfg['period_max']}" if use_sw else "NoSW"
        period_str_bw = f"{bw_cfg['period_min']}-{bw_cfg['period_max']}" if use_bw else "NoBW"
        sol_dir = fullpath(f"{mdir}/SOLUTIONS/UNIVERSAL/{engine}_{submodel}/{event_id}/{grid_type}/{ppa}_ppa/SW_{period_str_sw}_BW_{period_str_bw}/Z_MW/{depth_min}_{depth_max}_{depth_step}/{mw_init}_{mw_fin}_{mw_step}/")
        
        if not os.path.exists(sol_dir):
            os.makedirs(sol_dir)
            
        # Move output files and copy the JSON configuration file
        os.system(f"mv {event_id}{grid_type}* {sol_dir}")
        os.system(f"cp {config_file} {sol_dir}")

        # Execution tracking text file based exclusively on Python execution
        cmd_string = f"mpirun -np [NPROC] python universal_mtuq.py {config_file}"
        with open(f'{sol_dir}/command_mtuq.txt', 'w') as f:
            f.write(cmd_string)
            
        print(f'\nResults successfully saved to:\n{sol_dir}\n')
        print('Finished\n')