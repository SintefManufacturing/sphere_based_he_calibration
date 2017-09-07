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
import math3d as m3d

from hand_eye_calibration import ParkMartinCalibrator

from naming import scenere, scenetmpl, fibre, fibtmpl, oisre, oistmpl

# datafolder = 'data_' + datetime.date.today().isoformat()
datafolder = 'data_2017-09-06'
datafiles = os.listdir(datafolder)


oindices = []
findices = []
for fn in datafiles:
    om = oisre.match(fn)
    if om is not None:
        oindices.append(int(om.groups()[0]))
        continue
    fm = fibre.match(fn)
    if fm is not None:
        findices.append(int(fm.groups()[0]))

indices = list(set(oindices).intersection(findices))
if (len(indices) != len(findices) or
    len(indices) != len(oindices)):
    print('!!! WARNING !!! Mismatch between pose and scene indices')
indices.sort()

fibs = [m3d.Transform(np.loadtxt(os.path.join(datafolder, fibtmpl.format(i))))
        for i in indices]
oiss = [m3d.Transform(np.loadtxt(os.path.join(datafolder, oistmpl.format(i))))
        for i in indices]
sios = [t.inverse for t in oiss]

fib_sio_pairs = np.array(list(zip(fibs, sios)))

# Simple consensus based eviction
poses_minus = []
for i in range(len(fib_sio_pairs)):
    poses_minus.append(ParkMartinCalibrator(
        np.delete(fib_sio_pairs, i, axis=0)).sensor_in_flange.pose_vector)


# pmc = ParkMartinCalibrator(fib_sio_pairs)
# np.savetxt(os.path.join(datafolder, 'sensor_in_flange.npytxt'), pmc.sensor_in_flange.pose_vector)
