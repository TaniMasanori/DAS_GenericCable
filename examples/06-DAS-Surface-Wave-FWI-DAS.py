#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# In[2]:


import os
import numpy as np
import matplotlib.pyplot as plt
import torch

from seisfwi.model import ElasticModel
from seisfwi.survey import Survey, Source, Receiver, GenericCable, SeismicData
from seisfwi.propagator import ElasticPropagator
from seisfwi.problem import ElasticFWI, WaveformL2Loss

from seisfwi.utils import wavelet, moment_tensor, plot_data, plot_sg, timer, plot_misfit, load_misfit, smooth2d
from seisfwi.utils import generate_ellipse_anomaly, smart_cable


# ## Working path

# In[3]:


path = './37-FWI-DAS-NS'

os.makedirs(path, exist_ok=True)


# ### Parameters for elastic model

# In[4]:


ox = 0.0
oz = 0.0
nx = 481
nz = 81
dx = 6.25
dz = 6.25

nt = 4000
dt = 0.002
f0 = 2.0
amp = 1e7


# In[5]:


# True model for modeling obs data
vp_all = np.load('../models/Marmousi2/Marmousi-Vp-13601x2801-1.25m.npy').T
vs_all = np.load('../models/Marmousi2/Marmousi-Vs-13601x2801-1.25m.npy').T
rho_all = np.load('../models/Marmousi2/Marmousi-Rho-13601x2801-1.25m.npy').T

# cut the model
vp = vp_all[370:775:5, 3000:5405:5].astype(np.float32)
vs = vs_all[370:775:5, 3000:5405:5].astype(np.float32)
rho = rho_all[370:775:5, 3000:5405:5].astype(np.float32)


# In[6]:


model = ElasticModel(ox, oz, dx, dz, nx, nz, vp = vp, vs = vs, rho = rho, free_surface=True, nabc=20)  


# In[7]:


model


# In[8]:


cmap_range = {'vp': (1500, 1850), 'vs': (300, 550), 'rho': (1850, 2100)}
model.plot(cmap_range = cmap_range, pars = ['vs'], orientation='horizontal', aspect=1)


# ### Initial model

# In[9]:


nsmooth = 10
vp_int = smooth2d(vp, nsmooth, nsmooth)
vs_int = smooth2d(vs, nsmooth, nsmooth)
rho_int = smooth2d(rho, nsmooth, nsmooth)

# vp_int_1d = np.mean(vp_int, axis=1)
# vs_int_1d = np.mean(vs_int, axis=1)
# rho_int_1d = np.mean(rho_int, axis=1)

# # vp_int_1d = np.linspace(vp.min(), vp.max() * 0.9, num=nz)
# # vs_int_1d = np.linspace(vs.min(), vs.max() * 0.9, num=nz)
# # rho_int_1d = np.linspace(rho.min(), rho.max() * 0.9, num=nz)

# for i in range(nx):
#     vp_int[:,i] = vp_int_1d
#     vs_int[:,i] = vs_int_1d
#     rho_int[:,i] = rho_int_1d


# In[10]:


model_int = ElasticModel(ox, oz, dx, dz, nx, nz, vp = vp_int, vs = vs_int, rho = rho_int, free_surface=True, nabc=20)  


# In[11]:


model_int


# In[12]:


model_int.plot(cmap_range = cmap_range, pars = ['vs'], orientation='horizontal', aspect=1)


# ### Gradient Mask

# In[13]:


# set the gradient mask to exclude the water layers
nmask = 0
grad_mask = np.ones_like(vp)
grad_mask[:nmask, :] = 0
# grad_mask[nmask:nmask+10, :] = np.flip(np.repeat(np.linspace(0, 1, 10), nx).reshape(10, nx))


# ### Source

# In[14]:


# Create a seismic sources
source = Source(nt = nt, dt = dt, f0 = f0)

# Create wavelet
wvlt = wavelet(nt, dt, f0) * amp

# # Set source attributes
for isrc in range(23):
    source.add_source([125 * isrc + 125, dx], wvlt, 'vz')

source


# ## DAS Inversion

# In[15]:


# set trajectory
# traj = np.array([[750, 0.0, 30.0],[750, 0.0, 500.0], [750, 0.0, 1000.0],[850, 0.0, 1180.0], [1000, 0.0, 1300.0], [1100, 0.0, 1400.0]])

traj = np.array([[100, 0.0, dx],[2900, 0.0, dx], ])

# set the cable
chann_len = dx
chann_num = 446
gauge_len = dx

cable_surface = GenericCable(traj, chann_len, chann_num, gauge_len)
cable_surface

# Create a set of seismic receiver
das_surface = Receiver(nt = nt, dt = dt)

das_surface.add_cable(cable_surface)


# In[16]:


survey = Survey(source=source, receiver=das_surface, device='cuda', gpu_num=4, reciprocity=False, simultaneous=False, interpolation=False)


# In[17]:


survey.plot()


# In[18]:


F = ElasticPropagator(model, survey)


# In[19]:


dobs = F()


# In[20]:


dobs.plot(shotid=10, comp=['das'], clip=99.)


# In[21]:


model_fwi = ElasticModel(ox, oz, dx, dz, nx, nz,
                     vp = vp_int, 
                     vs = vs_int, 
                     rho = rho_int, 
                     vp_grad = False, 
                     vs_grad = True,
                     rho_grad = False,
                     vp_bound = [500, 2000],
                     vs_bound = [200, 600],
                     rho_bound = [1400, 2200],
                     free_surface=True, 
                     nabc=20,
                     mask = grad_mask)


# In[22]:


model_fwi.plot(cmap_range = cmap_range, pars = ['vs'], orientation='horizontal', aspect=1)


# In[23]:


model_int = ElasticModel.clone(model_fwi)
dsyn_int = F(model_int)


# In[24]:


dobs.plot_compare(dsyn_int, shotid=2, comp='das', clip=98, title=['OBS', 'SYN', 'Difference'], save_path=path + '/data-compare-int.png')


# In[25]:


dobs.plot_compare_trace(dsyn_int, shotid=2, comp='das', traceid=400, title=['OBS', 'SYN'], linestyle = ['k-', 'r--'], save_path = path + '/trace-int.png')


# In[26]:


model_fwi.plot_well_log()


# In[27]:


model_fwi.plot_mask()


# In[28]:


# Setup optimizer
optimizer = torch.optim.LBFGS(model_fwi.parameters(), lr = 1.0, max_iter = 50, tolerance_change = 1e-4, history_size = 5, line_search_fn = 'strong_wolfe')

# Setup misfit function
misfit = torch.nn.MSELoss()

# Setup components
weight = {'das': 1.0}


# In[29]:


fwi = ElasticFWI(F, model_fwi, optimizer, misfit, dobs, weight)


# In[30]:


fwi(nshot_per_gpu_per_batch = 8, max_iter = 2, grad_scale = 20.0, grad_interval=2)


# In[40]:


model_fwi.plot(grad=True, cmap='seismic', clip=99.0, pars = ['vs'],  aspect=1, orientation = 'horizontal',)


# In[41]:


model_fwi.plot(cmap_range = cmap_range, pars = ['vs'], orientation='horizontal', aspect=1)


# In[33]:


update_cmap_range = {'vp': (-1, 1), 'vs': (-60, 60), 'rho': (-20, 20)}

(model_fwi- model_int).plot(cmap_range = update_cmap_range, pars=['vs'], cmap='jet', orientation = 'horizontal')


# In[34]:


(model- model_int).plot(cmap_range = update_cmap_range, pars=['vs'], cmap='jet', orientation = 'horizontal')


# In[35]:


model_final = ElasticModel.clone(model_fwi)
dsyn_fwi = F(model_final)


# In[36]:


dobs.plot_compare(dsyn_fwi, shotid=2, comp='das', clip=98, title=['OBS', 'SYN', 'Difference'], save_path=path + '/data-compare-fwi.png')


# In[37]:


dobs.plot_compare_trace(dsyn_fwi, shotid=2, comp='das', traceid=300, title=['OBS', 'SYN'], linestyle = ['k-', 'r--'], save_path = path + '/trace-FWI.png')


# In[38]:


model_fwi.save(path + '/model_fwi_das')


# In[ ]:




