#!/usr/bin/env python
import json
import os
import sys
import numpy as np
import logging
from mtuq import read, download_greens_tensors, open_db
from mtuq.event import Origin, MomentTensor
from mtuq.util import fullpath
from mtuq.util.cap import parse_station_codes
from mtuq.util.signal import get_arrival, m_to_deg
from mpi4py import MPI
from obspy.clients.fdsn import Client
from obspy import UTCDateTime
from mtuq.util import AttribDict
from obspy import Stream
from pyproj import Geod
from obspy.geodetics import kilometer2degrees
from obspy.geodetics.base import gps2dist_azimuth
from obspy.taup import TauPyModel
from mtuq.util.math import to_mij, from_mij, to_v_w
from obspy import UTCDateTime
from mtuq import MomentTensor

# Set up logging
logging.basicConfig(level=logging.INFO)
logging.getLogger('matplotlib.mathtext').setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


error_codes = {
    1: "Database option not recognized",
    2: "Invalid input data.",
    3: "Network connection failed.",
    # ... more error codes ...
}


def read_config_file(json_path):
    """
    Reads the full configuration JSON.

    Returns:
        config (dict): A dictionary containing EVERY parameter from the input file.
    """
    
    # 1. READ THE FILE
    # This pulls in keys: data_path, weights_path, event_id, model, npts, 
    # n_stations, radius_km, noise_duration, sampling_rate, noise_scaling,
    # synthetic_duration, station_array, filter, etc.
    with open(json_path, 'r') as f:
        config = json.load(f)

    # -------------------------------------------------------
    # A. PROCESS TIME OBJECTS
    # -------------------------------------------------------
    # Convert Origin Time
    if 'origin' in config and 'time' in config['origin']:
        config['origin']['time'] = UTCDateTime(config['origin']['time'])
    
    # Convert Noise Start Time
    if 'noise_start' in config:
        config['noise_start'] = UTCDateTime(config['noise_start'])


    # -------------------------------------------------------
    # B. PROCESS SOURCE (MOMENT TENSOR)
    # -------------------------------------------------------
    # We take the 'source' dictionary and create a new key 'mt' 
    # that holds the actual MTUQ MomentTensor object.
    src = config['source']    
    
    # 1. Calculate Scalar Moment (M0) from Mw
    m0 = mw2mo(src['mw'])

    # 2. Calculate Tensor components (Mij) 
    if src['format'] == 'vw':
        print('Moment Tensor format: v-w')
        mij = to_mij(
            rho=m0*np.sqrt(2.),  # Tape2012,
            v=src['v'], 
            w=src['w'], 
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
    
    
    # 3. Create the object and store it in the config
    config['mt'] = MomentTensor(mij)


    # -------------------------------------------------------
    # C. VALIDATION (OPTIONAL PRINT)
    # -------------------------------------------------------
    # This ensures everything you need is actually there.
    print(f"Configuration Loaded Successfully from {json_path}")
    print(f"  > Event ID: {config.get('event_id')}")
    print(f"  > Model: {config.get('model')}")
    print(f"  > Stations: {config.get('n_stations')} (Array: {config.get('station_array')})")
    print(f"  > Origin Time: {config['origin']['time']}")
    print(f"  > Moment Tensor: {config['mt']}")

    return config
    
def fetch_noise(network, station, location, channel, starttime, duration, sampling_rate, output_units="DISP"):
    """
    Fetch a segment of seismic noise data from IRIS.

    Parameters:
    -----------
    network : str
        Network code of the station.
    station : str
        Station code.
    location : str
        Location code (often "00" or "").
    channel : str
        Channel code (e.g., "BHZ").
    starttime : str
        Start time of the noise segment in ISO format (e.g., "2009-04-07T19:00:00").
    duration : float
        Duration of the noise segment in seconds.
    sampling_rate : float
        Target sampling rate in Hz.
    output_units : str, optional
        Desired output units ("DISP" for displacement in meters, "VEL" for velocity in m/s).
        Default is "DISP".

    Returns:
    --------
    noise_data : np.ndarray
        Array containing the noise time series in the specified units.
    """
    client = Client("IRIS")
    start = UTCDateTime(starttime)
    end = start + duration

    try:
        # Fetch the waveform data
        st = client.get_waveforms(network, station, location, channel, start, end)

        # Fetch the instrument response
        inventory = client.get_stations(network=network, station=station, location=location,
                                        channel=channel, startbefore=start, endafter=end, level="response")

        # Remove instrument response to convert to physical units
        st.remove_response(inventory=inventory, output=output_units)

        # Apply basic preprocessing
        st.detrend("demean").detrend("linear")
        st.taper(max_percentage=0.05)
        st.resample(sampling_rate)

        return st[0].data

    except Exception as e:
        logger.warning(f"Could not fetch noise from {network}.{station}.{channel}: {e}")
        return np.zeros(int(duration * sampling_rate))
    
def assign_noise_to_synthetics(synthetics, real_noise, noise_scaling=1.5):
    """
    Assign different noise segments to each synthetic station.
    """
    num_stations = len(synthetics)
    noise_length = len(real_noise)
    segment_length = len(synthetics[0][0].data)
    step_size = noise_length // (num_stations + 1)
    
    for i, stream in enumerate(synthetics):
        noise_segment = real_noise[i * step_size : i * step_size + segment_length]
        if len(noise_segment) < segment_length:
            noise_segment = np.pad(noise_segment, (0, segment_length - len(noise_segment)))
        for trace in stream:
            noise_std = np.std(noise_segment)
            trace_std = np.std(trace.data)
            scale_factor = (trace_std / noise_std) * noise_scaling if noise_std > 0 else noise_scaling
            trace.data += noise_scaling * noise_segment
            logger.debug(f"Noise scaling factor for station {i}: {scale_factor}")
    return synthetics
    
def create_circular_stations(origin_lat, origin_lon, origin, n_stations, radius_km, 
                             network,sampling_rate, window_length_sec):
    """
    Create a list of synthetic stations placed evenly on a true circle 
    around a given latitude/longitude, using geodetic calculations.
    
    Parameters:
    -----------
    origin_lat : float
        Latitude of the event origin.
    origin_lon : float
        Longitude of the event origin.
    n_stations : int, optional
        Number of stations to create (default is 20).
    radius_km : float, optional
        Radius of the circle in kilometers (default is 50 km).
    network : str, optional
        Network code for synthetic stations (default is "SY").
    sampling_rate : float, optional
        Sampling rate in Hz for the stations (default is 50.0 Hz).
    window_length_sec : float, optional
        Length of the time window in seconds (default is 360 sec).
        
    Returns:
    --------
    stations : list
        List of synthetic Station objects.
    """
    from mtuq.station import Station

    geod = Geod(ellps="WGS84")  # Define Earth as an ellipsoid
    angles = np.linspace(0, 360, n_stations, endpoint=False)  # Uniform spacing

    stations = []
    delta = 1.0 / sampling_rate
    npts = int(window_length_sec * sampling_rate)

    for i, angle in enumerate(angles):
        # Compute new lat/lon using geodetic forward transformation
        station_lon, station_lat, _ = geod.fwd(origin_lon, origin_lat, angle, radius_km * 1000)

        station_code = f"STN{i:02d}"
        station_code = "ST{}d{}".format(i,radius_km)
        station = Station(
            latitude=station_lat,
            longitude=station_lon,
            depth_in_m=None,
            elevation_in_m=None,
            network=network,
            station=station_code,
            location="",
            sampling_rate=sampling_rate,
            delta=delta,
            starttime=origin['time'],
            endtime=origin['time']+window_length_sec,
            #starttime="2009-04-07T20:11:15.360003Z",
            #endtime="2009-04-07T20:17:55.320003Z",
            npts=npts,
            id=f"{network}.{station_code}.",
        )
        stations.append(station)
    
    return stations


def complete_sac_header(config,synthetics,reference_time,origin):
    """
    Completes the SAC header of synthetic traces with origin information.

    Args:
        origin (obspy.core.event.Origin): Origin object containing event information.
        synthetics (list of obspy.core.stream.Stream): List of streams containing synthetic traces.

    Returns:
        list of obspy.core.stream.Stream: List of streams with completed SAC headers.
    """

    logger.info('Creating SAC headers for the synthetic traces...')
    new_synthetics = []

    for station in synthetics:
        new_stream = Stream() # Create an empty stream for each station
        for trace in station:
            sac_attrib = AttribDict()
            sac_attrib['stla'] = trace.stats.latitude
            sac_attrib['stlo'] = trace.stats.longitude
            sac_attrib['stdp'] = 0.0
            sac_attrib['evla'] = config['origin']['latitude']
            sac_attrib['evlo'] = config['origin']['longitude']
            sac_attrib['evdp'] = config['origin']['depth_in_m']/1000  # Convert depth to km
            az = gps2dist_azimuth(sac_attrib['evla'],sac_attrib['evlo'],sac_attrib['stla'],sac_attrib['stlo'])
            sac_attrib['dist'] = az[0]/1000
            sac_attrib['az'] = az[1]
            sac_attrib['lovrok'] = 1
            sac_attrib['lcalda'] = 1
            sac_attrib['kevnm'] = trace.stats.location
            sac_attrib['kcmpnm'] = 'BH{}'.format(trace.stats.channel)
            sac_attrib['b'] = reference_time-origin['time']
            sac_attrib['o'] = 0
            trace.stats.starttime = reference_time
            trace.stats.channel = sac_attrib['kcmpnm']
            trace.stats.sac = sac_attrib #assign the sac attribdict to the trace.
            new_stream += trace #add the modified trace to the stream.

        new_synthetics.append(new_stream) #add the stream to the new_synthetics list.

    return new_synthetics

def save_synthetics(synthetics,config,path_mt='NA',suffix_dir=''):
    """
    Docstring for save_synthetics
    
    :param synthetics: synthetic seismograms to be saved. Obspy Stream object
    :param config: Configuration dictionary after reading the JSON file.
    :param path_mt: If you are calculating synthetics from a MTUQ MT solution you already inverted, provide the path to the MTUQ solution file here.
    :param suffix_dir: Tail to add to the synthetics directory name.
    """

    mt = config['mt']
    
    logger.info('Saving Synthetics...')
    
    model_and_database = config['model'].split('-')
    database = model_and_database[0]
    model  = model_and_database[1]

    if suffix_dir != '':
        dir_name = "synthetics/{}/{}/{}/{}_{}".format(database,model,config['station_array'],config['event_id'],suffix_dir)
    else:
        dir_name = "synthetics/{}/{}/{}/{}".format(database,model,config['station_array'],config['event_id'])

    os.system("mkdir -p {}".format(dir_name))

    for stream in synthetics:
        for trace in stream:
            name_trace = '{}.{}.{}.{}.{}.{}'.format(config['event_id'],trace.stats.network,trace.stats.station,trace.stats.location,
                                                    trace.stats.channel[0:-1],trace.stats.channel[-1].lower())
            print(name_trace)
            trace.write('{}/{}'.format(dir_name,name_trace),format='SAC')

    open_readme = open('{}/synthetics_info.txt'.format(dir_name),'w')
    mt_as_vector = mt.as_vector()
    open_readme.write('Moment Tensor used\n')
    open_readme.write('Mrr: {} \n'.format(mt_as_vector[0]))
    open_readme.write('Mtt: {} \n'.format(mt_as_vector[1]))
    open_readme.write('Mpp: {} \n'.format(mt_as_vector[2]))
    open_readme.write('Mrt: {} \n'.format(mt_as_vector[3]))
    open_readme.write('Mrp: {} \n'.format(mt_as_vector[4]))
    open_readme.write('Mtp: {} \n'.format(mt_as_vector[5]))
    open_readme.write('MTUQ SOLUTION PATH:\n')
    open_readme.write(path_mt)
    open_readme.write('\nBandpass Filter: {}'.format(config['filter']))
    open_readme.close()

    open_weights=open('{}/weights.dat'.format(dir_name),'w')
    open_weights.write("#  event_id.net.sta.loc.ch          offset_km    weights               P_pick  bw_len    S_pick  sw_len    rw_static  lw_static\n")
    for stream in synthetics:
        trace = stream[0]
        name_trace = '{}.{}.{}.{}.{}'.format(config['event_id'],trace.stats.network,trace.stats.station,trace.stats.location,trace.stats.channel[0:-1])
        line = '{}  {}   1 1 1 1 1      0   0   0   0   0   0\n'.format(name_trace,trace.stats.sac.dist)
        open_weights.write(line)
    open_weights.close()
    cp_config_file = 'cp config_synthetic.json {}/'.format(dir_name)
    print(cp_config_file)
    os.system(cp_config_file)
    logger.info('Synthetics saved on {}'.format(dir_name))

def calculate_starttime_synthetics(stations,config):

    logger.info('Calculating the earliest starttime based on the nearest station...')

    #model=config['model'].split('-')[1]
    #We use for default ak135 model. Given the purpose of this subroutine, make an start time  the velocity model is not too relevant
    model = 'ak135'
    model = TauPyModel(model=model)
    event_depth_km = config['origin']['depth_in_m']/1000

    evla = config['origin']['latitude']
    evlo = config['origin']['longitude']
    distances = []

    for station in stations:
        stla = station['latitude']
        stlo =  station['longitude']
        distance_in_m, azimuth, _ = gps2dist_azimuth(evla,evlo,stla,stlo)
        distances.append(distance_in_m)

    closest_distance = np.min(np.array(distances))

    arrivals = model.get_travel_times(source_depth_in_km=event_depth_km,distance_in_degree=m_to_deg(closest_distance),phase_list=['p','P'])

    logger.info('The closest station of the array is at {}km...'.format(closest_distance/1000))

    try:
        p_arrival = get_arrival(arrivals, 'p')
    except:
        p_arrival = get_arrival(arrivals, 'P')

    delta = p_arrival - 0.4*config['synthetic_duration']

    logger.info('The P-wave arrival is: {}s...'.format(p_arrival))
    logger.info('Earliest startime before(-) or after (+) the origin time: {}...'.format(delta))
    #logger.info('The synthetic duration will be : {}...'.format(delta))

    return delta

    
def run_synthetic_test(config):
    """
    Execute the synthetic test given a configuration dictionary.
    """
    # Unpack configuration parameters
    model_option = config['model'].split('-')
    database =  model_option[0]
    model = model_option[1]
    magnitude = config['source']['mw']  # Default to 4.5 if not specified
    event_id = config['event_id']
    
    # Define fixed origin
    origin = Origin(config['origin'])

    if  config['station_array'] == 'real':
        path_data = f"{config['event_id']}/*.[zrt]"
        path_weights = f"{config['event_id']}/weights.dat"
        
        logger.info('Reading data...')
        data = read(path_data, format='sac', event_id=event_id,
                    station_id_list=parse_station_codes(path_weights),
                    tags=[f'units:{config["units"]}', f'type:{config["type"]}'])
        data.sort_by_distance()

        stations = data.get_stations()
            
    elif  config['station_array'] == 'circle':
        path_data = fullpath(f"{config['event_id']}/*.[zrt]")
        path_weights = fullpath(f"{config['event_id']}/weights.dat")

        # Create synthetic station geometry (e.g., circular)
        stations = create_circular_stations(origin.latitude, origin.longitude,origin,
                                                config['circle_param']['n_stations'],config['circle_param']['radius_km'],config['circle_param']['circle_network'],
                                                config['sampling_rate'],config['synthetic_duration'])
        logger.info(f'Created {len(stations)} synthetic stations.')

    elif  config['station_array'] == 'manual':
        pass

    #Adjust starttime in stations.
    #If the starttime in stations is the origin time, the windowing in synthetics.map(process_bw) will yield this error:
    #Exception: The chosen window begins before the trace.  Consider using a later window, or to automatically pad the beginning of the trace with zeros, use mtuq.util.signal.resample instead
    #Therefore it is important to calculate the start time  to the synthetics as the origin time minus the windowing formula:
    #starttime = picks['P'] - 0.4*self.window_length
    #calculated for the clossest station. 
    shift_starttime  =  np.ceil(np.abs(calculate_starttime_synthetics(stations,config)))
    #This will be the reference time for all the synthetics. I give then extra seconds just in case. 
    reference_time = origin['time'] - shift_starttime - 10
    

    for station in stations:
        station['starttime'] = origin['time'] - shift_starttime - 10
        station['npts'] = station['npts'] + int((shift_starttime+10)*station['sampling_rate'])    

    if  database == 'syngine':  
        logger.info('Downloading Greens functions...')
        greens = download_greens_tensors(stations, origin, model)

    elif database == 'FK':
        logger.info('Reading Greens functions from FK database...')
        db = open_db('{}/{}'.format(database,model),format='FK')
        print('Reading Greens functions...\n')
        greens = db.get_greens_tensors(stations,origin)

    elif database == '3D':
        logger.info('Reading Greens functions from Specfem3D database...')
        db = open_db('{}/{}/'.format(database,model),format="SPECFEM3D")
        greens = db.get_greens_tensors(stations,origin)
        
    else:
        print('{}'.format(database))
        error_code = 1
        sys.exit(error_code)
    
    comm = MPI.COMM_WORLD
    if comm.rank == 0:    
     
        # Optionally, fetch real noise and add to synthetics

        #network, station, location, channel = nearest_station if nearest_station else ("IU", "COLA", "00", "BHZ")
        #logger.info(f"Using real noise from: {network}.{station}.{location}.{channel}")
        #real_noise = fetch_noise(network, station, location, channel,
        #                            config['noise_start'], config['noise_duration'], config['sampling_rate'])
            
        # Compute raw synthetics from greens functions using a selected source
        mt = config['mt']
        synthetics = greens.get_synthetics(mt, ['Z', 'R', 'T'])


        if config['noise_scaling'] > 0:
            logger.info('Adding noise to synthetics... TO BE IMPLEMENTED')
            #synthetics = assign_noise_to_synthetics(synthetics, real_noise, noise_scaling=config['noise_scaling'])
            
        # Assign metadata to each synthetic trace
        for i, stream in enumerate(synthetics):
            stream.station = stations[i]
            stream.origin = origin
            stream.tags = ['units:m', 'type:greens', 'type']

        synthetics = complete_sac_header(config,synthetics,reference_time,origin)
        
        for stream in synthetics:
            for trace in stream:
                trace.attrs = AttribDict({'weight': 1.0})
            
    else:
        stations = None
        synthetics = None

    stations = comm.bcast(stations, root=0)
    synthetics = comm.bcast(synthetics, root=0)

    return synthetics


def mw2mo(mw):
    #inverse of https://github.com/mtuqorg/mtuq/blob/0a09059cfb2dbf91a4ab9b41ff8b820b786e9d78/mtuq/event.py#L144
    exponent = (1.5*mw + 9.1)
    mo=np.power(10, exponent)
    return mo

if __name__ == '__main__':

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    # Read configuration file
    config = read_config_file('config_synthetic.json')
    # Calculate Synthetics
    synthetics = run_synthetic_test(config)
    #If the moment tensor comes from a MT solution obtained previously in MTUQ
    path_mt=config['annotations']['notes']
    suffix_dir=config['annotations']['suffix_dir']
    save_synthetics(synthetics,config,path_mt,suffix_dir)