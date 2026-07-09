# Code for processing SPECFEM3D_GLOBE elementary sources into an MTUQ-ready GF database.
# Engineered for robust, in-memory processing to preserve pristine simulation files.

import os
import glob
import json
import obspy
import argparse
from obspy.geodetics.base import gps2dist_azimuth
from obspy.signal.rotate import rotate_ne_rt

# Expected elementary sources
MT_COMPONENTS = ['MPP', 'MRP', 'MRR', 'MRT', 'MTP', 'MTT']

def load_config(config_path):
    """Loads the JSON configuration file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file {config_path} not found.")
    with open(config_path, 'r') as f:
        return json.load(f)

def parse_cmtsolution(cmt_path):
    """Parses event coordinates and MT scaling factors directly from CMTSOLUTION."""
    cmt_data = {}
    with open(cmt_path, 'r') as f:
        lines = f.readlines()
        
    for line in lines:
        if ':' in line:
            key = line.split(':')[0].strip().lower()
            if key == 'latitude':
                cmt_data['latitude'] = float(line.split(':')[1].strip())
            elif key == 'longitude':
                cmt_data['longitude'] = float(line.split(':')[1].strip())
                
        # Extract the Moment Tensor components
        tokens = line.split()
        if not tokens: continue
        first_token = tokens[0].split(':')[0].upper()
        
        if first_token in MT_COMPONENTS:
            try:
                # Get the scientific notation value and apply the scaling factor
                val = float(tokens[-1])
                cmt_data[first_token] = val / 1e7
            except ValueError:
                pass # Skips formatting quirks if MT values are split across lines
                
    return cmt_data

def validate_directories(base_solver_path):
    """Ensures all 6 MT directories exist and contain SAC files before processing."""
    for m in MT_COMPONENTS:
        mt_dir = os.path.join(base_solver_path, m, "OUTPUT_FILES")
        if not os.path.exists(mt_dir):
            print(f"FATAL ERROR: Directory missing - {mt_dir}")
            return False
        
        sac_files = glob.glob(os.path.join(mt_dir, "*sem.sac"))
        if len(sac_files) == 0:
            print(f"FATAL ERROR: No *sem.sac files found in {mt_dir}")
            return False
    return True

def process_event_depths(config_file):
    """Main orchestration function for processing GFs."""
    config = load_config(config_file)
    ev_id = config['event_id']
    v_model = config['velocity_model']
    procs = config['processors']
    depths = config['depths']

    print(f"=== Initiating MTUQ GF Processing Pipeline ===")
    print(f"Config File: {config_file}")
    print(f"Event: {ev_id} | Model: {v_model} | Partition: {procs}")

    for depth in depths:
        print(f"\n--- Processing Depth: {depth} km ---")
        base_solver_path = f"../SIMULATIONS/{v_model}/SOLVER_REPO/{procs}/{ev_id}/{depth}"
        out_dir = f"../READY_GFs/{v_model}/{procs}/{ev_id}/{depth}"
        
        # 1. Gatekeeper Check
        if not validate_directories(base_solver_path):
            print(f"Skipping depth {depth} due to missing files or directories.")
            continue

        # 2. Extract Event Coordinates from MPP (Coordinates remain constant across all sources)
        master_cmt_path = os.path.join(base_solver_path, 'MPP', 'DATA', 'CMTSOLUTION')
        if not os.path.exists(master_cmt_path):
            print(f"Cannot find {master_cmt_path}. Skipping depth.")
            continue
            
        master_cmt_data = parse_cmtsolution(master_cmt_path)
        ev_lat = master_cmt_data.get('latitude')
        ev_lon = master_cmt_data.get('longitude')
        
        # 3. Create pristine output directory
        os.makedirs(out_dir, exist_ok=True)
        print(f"Output directory established: {out_dir}")

        # 4. Process each elementary source
        for m in MT_COMPONENTS:
            print(f"  > Processing {m}...")
            
            # Fetch the scale factor dynamically from THIS component's CMTSOLUTION
            comp_cmt_path = os.path.join(base_solver_path, m, 'DATA', 'CMTSOLUTION')
            if not os.path.exists(comp_cmt_path):
                print(f"    ! ERROR: Cannot find {comp_cmt_path}. Skipping component.")
                continue

            comp_cmt_data = parse_cmtsolution(comp_cmt_path)
            scale_factor = comp_cmt_data.get(m)
            
            # Safety check to prevent divide-by-zero crashes
            if scale_factor == 0.0 or scale_factor is None:
                print(f"    ! WARNING: Scale factor for {m} is 0.0 or missing. Check {comp_cmt_path}. Skipping rotation.")
                continue 

            m_short = f"M{m[1:3].lower()}" # e.g., 'Mpp'
            
            sac_files = glob.glob(f"{base_solver_path}/{m}/OUTPUT_FILES/*sem.sac")
            
            # Group files by network.station to handle 3-component rotation efficiently
            stations = {}
            for f in sac_files:
                basename = os.path.basename(f)
                parts = basename.split('.')
                net = parts[0]
                sta = parts[1]
                comp = parts[2][-1] # Grabs X, Y, Z, E, or N
                
                key = f"{net}.{sta}"
                if key not in stations:
                    stations[key] = {}
                stations[key][comp] = f
                
            # Process rotation and scaling purely in memory
            for key, comps in stations.items():
                if 'Z' not in comps or not (('E' in comps and 'N' in comps) or ('X' in comps and 'Y' in comps)):
                    continue # Skip incomplete stations silently to maintain speed
                    
                # Read raw files
                st_Z = obspy.read(comps['Z'])[0]
                if 'X' in comps:
                    st_E = obspy.read(comps['X'])[0]
                    st_N = obspy.read(comps['Y'])[0]
                else:
                    st_E = obspy.read(comps['E'])[0]
                    st_N = obspy.read(comps['N'])[0]

                # Apply scale factor directly in memory
                st_Z.data = st_Z.data / scale_factor
                st_E.data = st_E.data / scale_factor
                st_N.data = st_N.data / scale_factor

                # Extract station coordinates dynamically from header
                st_lat = st_Z.stats.sac.stla
                st_lon = st_Z.stats.sac.stlo
                net = st_Z.stats.network
                sta = st_Z.stats.station

                # Calculate Back-Azimuth and Rotate
                # Coordinate order MATTERS: Station Point 1 -> Event Point 2
                baz = gps2dist_azimuth(st_lat, st_lon, ev_lat, ev_lon)
                
                # baz[1] = Back-Azimuth (Station pointing toward Event)
                # baz[2] = Forward Azimuth (Event pointing toward Station)
                rotated_r, rotated_t = rotate_ne_rt(st_N.data, st_E.data, baz[1])

                # Clean headers and output finalized files
                # Z Component
                st_Z.stats.sac.khole = ''
                st_Z.stats.location = ''
                st_Z.stats.sac.kcmpnm = 'BHZ'
                st_Z.stats.channel = 'BHZ'
                st_Z.write(f"{out_dir}/{net}.{sta}..Z.{m_short}.sac", format='SAC')

                # R Component
                st_R = st_N.copy()
                st_R.data = rotated_r
                st_R.stats.sac.kcmpnm = 'BHR'
                st_R.stats.channel = 'BHR'
                st_R.stats.sac.cmpaz = baz[2]
                st_R.stats.sac.khole = ''
                st_R.stats.location = ''
                st_R.write(f"{out_dir}/{net}.{sta}..R.{m_short}.sac", format='SAC')

                # T Component
                st_T = st_E.copy()
                st_T.data = rotated_t
                st_T.stats.sac.kcmpnm = 'BHT'
                st_T.stats.channel = 'BHT'
                st_T.stats.sac.cmpaz = (baz[2] + 90) % 360
                st_T.stats.sac.khole = ''
                st_T.stats.location = ''
                st_T.write(f"{out_dir}/{net}.{sta}..T.{m_short}.sac", format='SAC')

    print("\n=== Pipeline Complete ===")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process SPECFEM3D_GLOBE GFs into an MTUQ-ready database.")
    parser.add_argument("config_file", help="Path to the JSON configuration file (e.g., config_20171201023244.json)")
    args = parser.parse_args()
    
    process_event_depths(args.config_file)