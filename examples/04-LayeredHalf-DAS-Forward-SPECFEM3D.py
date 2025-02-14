#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Converted from 04-LayeredHalf-DAS-Forward-SPECFEM3D.ipynb
"""

# %% Cell 0: Autoreload (Jupyter magics commented out)
# %load_ext autoreload
# %autoreload 2

# %% Cell 1: Matplotlib inline (Jupyter magic commented out)
# %matplotlib inline

# %% Cell 2: SPECFEM3D installation instructions
# Install SPECFEM3D:
# Please refer to: https://github.com/SPECFEM/specfem3d

# %% Cell 3: Import libraries
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
from IPython.display import Image

from scipy.interpolate import griddata, splprep, splev
from scipy.ndimage import gaussian_filter
from scipy import signal

import utm

# %% Cell 4: Load topography
topo = np.loadtxt('/home/masa/specfem3d/EXAMPLES/applications/layered_halfspace/ptopo.mean.utm')

# Extract individual columns
lon = topo[:, 0]
lat = topo[:, 1]
elev = topo[:, 2]

# Creating a grid to interpolate
lons = np.linspace(np.min(lon), np.max(lon), 200)
lats = np.linspace(np.min(lat), np.max(lat), 200)

grid_lon, grid_lat = np.meshgrid(lons, lats)
grid_elev = griddata((lon, lat), elev, (grid_lon, grid_lat), method='cubic')

# Remove NaNs at the boundary using slicing
grid_elev[:5, :] = grid_elev[5, None, :]
grid_elev[-5:, :] = grid_elev[-5, None, :]
grid_elev[:, :5] = grid_elev[:, 5, None]
grid_elev[:, -5:] = grid_elev[:, -5, None]

# %% Cell 5: (Plot topography removed)

# %% Cell 6: Design the fiber cable trajectory
cable = []
max_elevation_idx = np.argmax(elev)
# cable.append([lon[max_elevation_idx-1], lat[max_elevation_idx-1]])
cable.append([lons[50*2], lats[45*2]])
cable.append([lons[47*2], lats[42*2]])
cable.append([lons[44*2], lats[40*2]])
cable.append([lons[40*2], lats[35*2]])
cable.append([lons[42*2], lats[30*2]])
cable.append([lons[48*2], lats[28*2]])
cable.append([lons[55*2], lats[27*2]])
cable.append([lons[60*2], lats[26*2]])
cable.append([lons[65*2], lats[25*2]])
cable.append([lons[70*2], lats[22*2]])
cable.append([lons[75*2], lats[19*2]])
cable.append([lons[80*2], lats[19*2]])
cable.append([lons[85*2], lats[20*2]])
cable.append([lons[90*2], lats[22*2]])
cable.append([lons[95*2], lats[24*2]])
# cable.append([lons[98*2], lats[26*2]])
cable = np.array(cable)

# %% Cell 7: Spline interpolation for cable trajectory
num_points = 2500

# Prepare data for spline fitting (using all coordinates)
tck, _ = splprep([cable[:, 0], cable[:, 1]], s=0)

# Generate new, smoothed points from the spline representation
cable_smoothed = splev(np.linspace(0, 1, num_points), tck)
cable_smoothed = np.array(cable_smoothed).T

# %% Cell 8: (Plot of the cable trajectory over topography removed)

# %% Cell 9: Compute the cable elevation
cable_elev = griddata((lon, lat), elev, (cable_smoothed[:, 0], cable_smoothed[:, 1]), method='cubic')
cable_coord = np.concatenate((cable_smoothed, cable_elev.reshape(-1, 1)), axis=1)

# %% Cell 10: (Plot cable elevation removed)

# %% Cell 11: (3D topography with fiber cable trajectory plot removed)

# %% Cell 12: Setup the Generic Cable object
from genericcable import GenericCable

traj = cable_coord.copy()
traj[:, 2] = -traj[:, 2]  # Invert the elevation
chann_len = 20.0         # Channel spacing (m)
chann_num = 679          # Number of channels (set initially)
gauge_len = 20.0         # Gauge length (m)

cable = GenericCable(traj, chann_len, chann_num, gauge_len)

# %% Cell 13: Print cable information
print(cable)

# %% Cell 14: Get channel coordinates from the cable
chan_coord = cable.get_chann_coords()
print("Channel coordinates shape:", chan_coord.shape)

# %% Cell 15: Smooth the sensitivity vectors
cable.tan_cha[:, 0] = gaussian_filter(cable.tan_cha[:, 0], sigma=2)
cable.tan_cha[:, 1] = gaussian_filter(cable.tan_cha[:, 1], sigma=2)
cable.tan_cha[:, 2] = gaussian_filter(cable.tan_cha[:, 2], sigma=2)


# %% Cell 23: Cable dot product test
cable.dot_product_test()

# %% Cell 24: Get receiver locations and convert UTM to lat/lon
rec_loc = cable.get_rec_loc_unique()

rec_loc_lonlat = []
for i in range(rec_loc.shape[0]):
    # Convert from UTM (zone 10T) to lat/lon
    rec_loc_lonlat.append(utm.to_latlon(rec_loc[i, 0], rec_loc[i, 1], 10, zone_letter='T'))
rec_loc_lonlat = np.array(rec_loc_lonlat)

# (Receiver location plotting removed)

# %% Cell 25: Write the station file for SPECFEM3D
with open('/home/masa/specfem3d/EXAMPLES/applications/Mount_StHelens/DATA/STATIONS', 'w') as file:
    for i in range(rec_loc_lonlat.shape[0]):
        lon_val = rec_loc_lonlat[i, 0]
        lat_val = rec_loc_lonlat[i, 1]
        file.write(f"STA{i+1:03} \t DAS \t {lon_val:.4f} \t {lat_val:.4f} \t 0.0 \t 0.0\n")

# %% Cell 26: Load and process a SPECFEM3D receiver VTK file
def load_vtk_file(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()
    
    # Find the start of the points data
    start_idx = 0
    for i, line in enumerate(lines):
        if line.strip().startswith('POINTS'):
            start_idx = i + 1
            break
    start_idx += 1  # Skip source location
    points_data = lines[start_idx:-1]
    points = [list(map(float, line.split())) for line in points_data]
    return np.array(points)

rec_specfem = load_vtk_file("/home/masa/specfem3d/EXAMPLES/applications/layered_halfspace/OUTPUT_FILES/sr.vtk")
print("rec_specfem shape:", rec_specfem.shape)

# %% Cell 30: Load DAS waveform data
nrec = rec_loc.shape[0]
data_vx = []
data_vy = []
data_vz = []

for i in range(nrec):
    vx = np.loadtxt(f"/home/masa/specfem3d/EXAMPLES/applications/Mount_StHelens/OUTPUT_FILES/DAS.STA{i+1:03}.HXE.semv")
    vy = np.loadtxt(f"/home/masa/specfem3d/EXAMPLES/applications/Mount_StHelens/OUTPUT_FILES/DAS.STA{i+1:03}.HXN.semv")
    vz = np.loadtxt(f"/home/masa/specfem3d/EXAMPLES/applications/Mount_StHelens/OUTPUT_FILES/DAS.STA{i+1:03}.HXZ.semv")
    tt = vz[:, 0]
    data_vx.append(vx[:, 1])
    data_vy.append(vy[:, 1])
    data_vz.append(vz[:, 1])

t = tt - tt[0]
data_vel = np.array([data_vx, data_vy, data_vz])

# %% Cell 31: (DAS waveform component imshow plots removed)

# %% Cell 32: Compute forward DAS data
data_das = cable.forward(data_vel, m_comp='vel', d_comp='strain_rate')
# (DAS waveform plotting removed)


# %% Cell 37: Print time vector details
print("Time vector t:", t)
