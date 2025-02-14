#!/usr/bin/env python
# coding: utf-8

# # DAS Cable

# In[1]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# In[1]:


import numpy as np
import matplotlib.pyplot as plt

from genericcable import GenericCable


# In[2]:


font_size = 12
params = {
    'image.cmap': 'seismic',
    'axes.grid': False,
    'savefig.dpi': 300,
    'font.size': font_size,
    'axes.labelsize': font_size,
    'legend.fontsize': font_size,
    'xtick.labelsize': font_size,
    'ytick.labelsize': font_size,
}

plt.rcParams.update(params)


# ## Cable 1

# In[3]:


# Define parameters
A = 150.0  # Radius of the spring
B = 2  # Pitch or vertical spacing between turns

# Create values for the parameter t
t = np.linspace(0, 13 * np.pi, 40)

# Calculate the coordinates
x = A * np.cos(0.5*t)
z = A * np.sin(0.5*t)
y = B * t * 20

traj = np.column_stack((x, y, z))

chann_len = 40.0
chann_num = 75
gauge_len = 10.0

cable = GenericCable(traj, chann_len, chann_num, gauge_len)


# In[4]:


cable


# In[6]:


cable.plot_traj()


# In[5]:


cable.plot_channel(show_tangent=True, save_path='cable2.png')


# ## Complex Cable 2

# Reference: 
# 
# White, D., Bellefleur, G., Dodds, K., & Movahedzadeh, Z. (2022). Toward improved distributed acoustic sensing sensitivity for surface-based reflection seismics: Configuration tests at the Aquistore CO2 storage site. Geophysics, 87(2), P1-P14.

# In[7]:


# Define parameters
A = 20.0  # Radius of the spring
B = 2     # Pitch or vertical spacing between turns

# Create values for the parameter t
t = np.linspace(0, 13 * np.pi, 40)

# Calculate the coordinates
x = A * np.cos(0.5*t)
y = A * np.sin(0.5*t)
z = B * t * 20

traj1 = np.column_stack((x, y, z))
traj2 = np.array([[5000., 0, 0.0], [2800., 0, 0], [2800., 0, 1400], [2400., 0, 1400], [2400., 0, 0], [1200., 0, 0], [160., 0, 0]])
traj3 = np.array([[0., 0, 1600], [0., 0, 12], [-1800., 0, 0], [-1800., 0, 1400], [-2200., 0, 1400], [-2200., 0, 0], [-5000., 0, 0]])

traj = np.vstack((traj2, traj1, traj3))


# In[8]:


chann_len = 50.0
chann_num = 1800
gauge_len = 50.0

cable = GenericCable(traj, chann_len, chann_num, gauge_len)


# In[9]:


cable.plot_traj()


# In[10]:


cable.plot_channel(show_tangent=True, save_path='cable3.png')


# In[11]:


cable.plot_sensitivity()


# In[ ]:




