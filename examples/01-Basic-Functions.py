#!/usr/bin/env python
# coding: utf-8

# # Basic Functions for DAS Cable

# In[1]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# In[2]:


import numpy as np
import matplotlib.pyplot as plt

from genericcable import GenericCable


# In[3]:


font_size = 12
params = {
    'image.cmap': 'seismic',
    'axes.grid': False,
    'savefig.dpi': 300,
    'font.size': font_size,
    'axes.labelsize': font_size,
    'axes.titlesize': font_size,
    'legend.fontsize': font_size,
    'xtick.labelsize': font_size,
    'ytick.labelsize': font_size
}
plt.rcParams.update(params)


# ### Set cable and analysis the cable

# In[4]:


traj = np.array([[100., 0, 100.0], [200., 0, 600.0], [500., 0, 1000.0], [1500., 0, 1000.0]])

# set large length for better plotting
chann_len = 150.0
chann_num = 130
gauge_len = 10.0

cable = GenericCable(traj, chann_len, chann_num, gauge_len)


# In[5]:


cable


# In[6]:


cable.dot_product_test()


# In[7]:


cable.plot_traj()


# In[8]:


cable.plot_channel


# In[9]:


cable.plot_channel(show_tangent=True, save_path='cable1.png')


# In[10]:


cable.plot_sensitivity()


# ### How to use in forward modeling and inversion

# In[11]:


traj = np.array([[100., 0, 100.0], [200., 0, 600.0], [500., 0, 1000.0], [1500., 0, 1000.0]])

# set normal length that could be used in real life
chann_len = 10.0
chann_num = 199
gauge_len = 10.0

cable = GenericCable(traj, chann_len, chann_num, gauge_len)


# In[12]:


cable


# In[13]:


cable.dot_product_test()


# In[14]:


cable.plot_traj()


# In[15]:


cable.plot_channel()


# In[16]:


cable.plot_sensitivity()


# ### Obtain the receiver location for exsisting propagator

# In[17]:


rec_loc = cable.get_rec_loc_unique()


# In[18]:


rec_loc.shape


# ### Run your propagator with above 3-component geophone locations

# In[19]:


# load the 3C geophone waveforms here
nt = 201
dt = 0.002
nrec = rec_loc.shape[0]
geo_data = np.random.rand(3, nrec, nt)


# ### DAS data forward modeling

# In[20]:


das_data = cable.forward(
             geo_data, 
             m_comp = 'vel', 
             d_comp = 'strain',   #'strain_rate'
             dt = dt)


# In[21]:


das_data.shape


# ### DAS data adjoint modeling for inversion

# In[22]:


# let's say we compare the syn and obs DAS data and have some residual 
das_res = np.random.rand(199, nt)


# In[23]:


# run the adjoint operator to get the residual in the geophone data space
geo_res = cable.adjoint(
            das_res, 
            m_comp = 'vel', 
            d_comp = 'strain',   #'strain_rate'
            dt = dt)


# In[24]:


geo_res.shape


# ### Then, you can provide the geophone residual to your FWI code based on the geophone data ( particle velocity component in this case)

# In[ ]:




