# coding=utf-8

"""
"""

__author__ = "Morten Lind"
__copyright__ = "SINTEF 2017"
__credits__ = ["Morten Lind"]
__license__ = "GPLv3"
__maintainer__ = "Morten Lind"
__email__ = "morten.lind@sintef.no"
__status__ = "Development"

import os
import datetime
import re

import numpy as np
import pcl

from sphere_recognition import find_object
from naming import scenere, scenetmpl, fibre, fibtmpl, oisre, oistmpl

# datafolder = 'data_' + datetime.date.today().isoformat()
datafolder = 'data_2017-09-06'
datafiles = os.listdir(datafolder)

obj_dims = (0.15, 0.125)
ball_radius = 0.037

sindices = []
findices = []
for fn in datafiles:
    sm = scenere.match(fn)
    if sm is not None:
        sindices.append(int(sm.groups()[0]))
        continue
    fm = fibre.match(fn)
    if fm is not None:
        findices.append(int(fm.groups()[0]))

indices = list(set(sindices).intersection(findices))
if (len(indices) != len(findices) or
    len(indices) != len(sindices)):
    print('!!! WARNING !!! Mismatch between pose and scene indices')
indices.sort()

scenes = []
fibs = []
oiss = []
valid_indices = []
for i in indices:
    oisfname = oistmpl.format(i)
    if oisfname in datafiles:
        print('Object transform exists for index {}, skipping.'.format(i))
        valid_indices.append(i)
        continue
    scene = pcl.load(os.path.join(datafolder, scenetmpl.format(i)))
    ois = find_object(scene, obj_dims, ball_radius)
    if ois is not None:
        np.savetxt(os.path.join(datafolder, oisfname), ois.pose_vector)
        scenes.append(scene)
        oiss.append(ois)
        fibs.append(np.loadtxt(os.path.join(datafolder, fibtmpl.format(i))))
        valid_indices.append(i)
