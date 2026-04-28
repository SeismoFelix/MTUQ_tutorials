#!/usr/bin/env python

import os
import sys
import json
import argparse
import numpy as np

from mtuq import read, open_db
from mtuq.event import Origin
from mtuq.graphics import plot_data_greens1, plot_misfit_depth, plot_misfit_dc, plot_beachball, plot_misfit_lune,\
    plot_variance_reduction_lune, plot_variance_reduction_dc
from mtuq.grid import DeviatoricGridSemiregular
from mtuq.grid_search import grid_search
from mtuq.misfit.waveform import Misfit, calculate_norm_data
from mtuq.process_data import ProcessData
from mtuq.util import fullpath, merge_dicts, save_json
from mtuq.util.cap import parse_station_codes, Trapezoid
# Warning: mpi4py must be imported before other heavy libraries in some environments,
# but MTUQ usually handles this. Ensuring it's available.
from mpi4py import MPI


class Header:
    def __init__(self, station, component, time_shift, cc):
        self.station = station
        self.component = component
        self.time_shift = time_shift
        self.cc = cc

def parse_args():
    parser = argparse.ArgumentParser(
        description="Run MTUQ Grid Search using a parameters_inversion.json file.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("-event", type=str, required=True, 
                        help="Event ID (directory must exist in current path e.g., -event 20220731...)")
    return parser.parse_args()

def load_params(event_id):
    """Loads parameters_inversion.json from the event directory."""
    params_path = os.path.join(os.getcwd(), event_id, 'parameters_inversion.json')
    if not os.path.exists(params_path):
        print(f"Error: Configuration file not found at {params_path}")
        sys.exit(1)
    
    with open(params_path, 'r') as f:
        return json.load(f)

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
            # Using f-string for cleaner output
            print(f"{stations[_i]['station']},{component}: {np.round(time_shift, 2)} {np.round(max_cc, 2)}")

    return header_info

def wrap_up(ts_list, cc_list, station):
    # Calculate averages ignoring NaNs
    valid_ts_count = ts_list.size - np.count_nonzero(np.isnan(ts_list))
    total_ts = np.round(np.nansum(np.abs(ts_list)) / valid_ts_count, 2) if valid_ts_count > 0 else np.nan
    
    valid_cc_count = cc_list.size - np.count_nonzero(np.isnan(cc_list))
    total_cc = np.round(np.nansum(cc_list) / valid_cc_count, 2) if valid_cc_count > 0 else np.nan

    line = f"{station} {ts_list[0]} {ts_list[1]} {ts_list[2]} {total_ts} {cc_list[0]} {cc_list[1]} {cc_list[2]} {total_cc}"
    return line

def write_headers(header_info, event_id, tag):
    filename = f'{event_id}DEV+Z_header_info_{tag}.txt'
    with open(filename, 'w') as open_header_file:
        open_header_file.write('STATIONS  ts_Z ts_R ts_T abs_av_shift cc_Z cc_R cc_T av_cc\n')

        init_stat = header_info[0].station
        ts_list = np.array([np.nan, np.nan, np.nan])
        cc_list = np.array([np.nan, np.nan, np.nan])

        for i in range(len(header_info)):
            if header_info[i].station != init_stat:
                line = wrap_up(ts_list, cc_list, init_stat)
                open_header_file.write(line + '\n')
                print(line)

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
                    print(line)

if __name__ == '__main__':
    # Initialize MPI
    comm = MPI.COMM_WORLD

    # Parse Event ID
    args = parse_args()
    event_id = args.event
    mdir = os.getcwd()

    if comm.rank == 0:
        print(f"Running MTUQ for event: {event_id}")

    # Load parameters from JSON
    params = load_params(event_id)

    # Paths
    path_data = fullpath(f'{mdir}/{event_id}/*.[zrt]')
    path_weights = fullpath(f'{mdir}/{event_id}/weights.dat')
    
    # Database setup - using IR database
    #model = 'ir'
    model = 'iran_ak135'
    db = open_db(f'{mdir}/FK/ir', format='FK')

    # Data Processing Settings (Surface Waves Only)
    # Params now expected to contain explicit period min/max
    period_min = float(params['period_min'])
    period_max = float(params['period_max'])
    window_len = float(params['window_length'])

    process_sw = ProcessData(
        filter_type='Bandpass',
        freq_min=1.0 / period_max,
        freq_max=1.0 / period_min,
        pick_type='FK_metadata',
        FK_database=f'{mdir}/FK/ir',
        window_type='surface_wave',
        window_length=window_len,
        capuaf_file=path_weights,
    )

    # Misfit Settings
    # Time shifts are now parameterized
    misfit_sw = Misfit(
        norm='L2',
        time_shift_min=float(params['time_shift_min']),
        time_shift_max=float(params['time_shift_max']),
        time_shift_groups=['ZR', 'T'],
    )

    # Station weights
    station_id_list = parse_station_codes(path_weights)

    # Magnitude Grid
    mw_init = float(params['mw_init'])
    mw_fin = float(params['mw_fin'])
    mw_step = float(params['mw_step'])
    # Ensure the final value is included
    magnitudes = np.arange(mw_init, mw_fin + mw_step/10.0, mw_step)

    # Depth Grid
    depth_min = float(params['depth_min'])
    depth_max = float(params['depth_max'])
    depth_step = float(params['depth_step'])
    
    # Convert km to meters for PREM/MTUQ
    prem_depths = np.arange(depth_min * 1000, (depth_max * 1000) + (depth_step * 1000)/10.0, depth_step * 1000)

    depths = []
    for d in prem_depths:
        # Sanity check for depths (optional, based on your previous code)
        if d < 1000 or d > 19000:
            pass
        else:
            depths.append(d)
    depths = np.array(depths)
    
    if comm.rank == 0:
        print(f"Depths (m): {depths}")
        print(f"Magnitudes: {magnitudes}")

    # Origin Definition
    catalog_origin = Origin({
        'time': params['origin_time'],
        'latitude': float(params['latitude']),
        'longitude': float(params['longitude']),
        'depth_in_m': np.average(depths) if len(depths) > 0 else 0,
    })

    origins = []
    for depth in depths:
        origins += [catalog_origin.copy()]
        setattr(origins[-1], 'depth_in_m', depth)

    # Grid and Wavelet
    grid = DeviatoricGridSemiregular(
        npts_per_axis=int(params['npts_per_axis']),
        magnitudes=magnitudes
    )

    wavelet = Trapezoid(
        magnitude=np.average(magnitudes)
    )

    #
    # The main I/O work starts now
    #
    if comm.rank == 0:
        print('Reading data...\n')
        data = read(path_data, format='sac',
                    event_id=event_id,
                    station_id_list=station_id_list,
                    tags=['units:m', 'type:velocity'])

        data.sort_by_distance()
        stations = data.get_stations()

        print('Processing data...\n')
        data_sw = data.map(process_sw)

        print('Reading Greens functions...\n')
        greens = db.get_greens_tensors(stations, origins)

        print('Processing Greens functions...\n')
        greens.convolve(wavelet)
        greens_sw = greens.map(process_sw)

    else:
        stations = None
        data_sw = None
        greens_sw = None

    # Broadcast data to all workers
    stations = comm.bcast(stations, root=0)
    data_sw = comm.bcast(data_sw, root=0)
    greens_sw = comm.bcast(greens_sw, root=0)

    #
    # The main computational work starts now
    #
    if comm.rank == 0:
        print('Evaluating surface wave misfit...\n')

    results_sw = grid_search(
        data_sw, greens_sw, misfit_sw, origins, grid)

    if comm.rank == 0:
        results = results_sw

        # Collect best-fitting source info
        origin_idx = results.origin_idxmin()
        best_origin = origins[origin_idx]

        source_idx = results.source_idxmin()
        best_mt = grid.get(source_idx)
        lune_dict = grid.get_dict(source_idx)
        mt_dict = best_mt.as_dict()

        merged_dict = merge_dicts(
            mt_dict, lune_dict, {'M0': best_mt.moment()},
            {'Mw': best_mt.magnitude()}, best_origin)

        # Generate figures and save results
        print('Generating figures...\n')

        plot_data_greens1(f"{event_id}DEV+Z_waveforms.pdf",
                          data_sw, greens_sw, process_sw,
                          misfit_sw, stations, best_origin, best_mt, lune_dict)

        if len(depths) > 1:
            plot_misfit_depth(f"{event_id}DEV+Z_misfit_depth.pdf", results, origins,
                              title=event_id)
            plot_misfit_depth(f"{event_id}DEV+Z_misfit_depth_tradeoffs.pdf", results, origins,
                              show_tradeoffs=True, show_magnitudes=True, title=event_id)

        plot_beachball(f"{event_id}DEV+Z_beachball.pdf",
                       best_mt, stations, best_origin,taup_model=model)

        print('Saving results...\n')

        header_info_sw = get_headerinfo(data_sw, greens_sw, misfit_sw, stations, best_origin, best_mt)
        write_headers(header_info_sw, event_id, 'sw')

        save_json(f"{event_id}DEV+Z_solution.json", merged_dict)

        origins_dict = {_i: origin for _i, origin in enumerate(origins)}
        save_json(f"{event_id}DEV+Z_origins.json", origins_dict)

        results.save(f"{event_id}DEV+Z_misfit.nc")

        plot_misfit_dc(f"{event_id}DEV+Z_DC_misfit.pdf", results)
        plot_misfit_lune(f"{event_id}DEV+Z_misfit.pdf", results)
        plot_misfit_lune(f"{event_id}DEV+Z_misfit_mt.pdf", results, show_mt=True)
        plot_misfit_lune(f"{event_id}DEV+Z_misfit_tradeoff.pdf", results, show_tradeoffs=True)

        norm_sw = calculate_norm_data(data_sw, misfit_sw.norm, ['Z', 'R', 'T'])
        plot_variance_reduction_lune(f"{event_id}DEV+Z_variance_reduction.pdf", results, norm_sw,
                                     colorbar_label='Variance reduction (percent)', show_mt=True)
        plot_variance_reduction_dc(f"{event_id}DEV+Z_DC_variance_reduction.pdf", results, norm_sw,
                                   colorbar_label='Variance reduction (percent)')

        print('\nFinished\n')