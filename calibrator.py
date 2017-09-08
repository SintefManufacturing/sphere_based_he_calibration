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

np.set_printoptions(precision=3)

pmc = ParkMartinCalibrator(fib_sio_pairs)
np.savetxt(os.path.join(datafolder, 'sensor_in_flange.npytxt'), pmc.sensor_in_flange.pose_vector)
print(pmc.sensor_in_flange)

# Simple consensus based eviction
poses_minus = []
for i in range(len(fib_sio_pairs)):
    poses_minus.append(ParkMartinCalibrator(
        np.delete(fib_sio_pairs, i, axis=0)).sensor_in_flange.pose_vector)
poses_minus = np.array(poses_minus)
print(poses_minus)

pos = poses_minus[:,:3]
pos_avg = np.average(pos, axis=0)
pos_devv = pos - pos_avg
pos_dev = np.linalg.norm(pos_devv, axis=1)

rot = poses_minus[:,3:]
rot_avg = np.average(rot, axis=0)
rot_devv = rot - rot_avg
rot_dev = np.linalg.norm(rot_devv, axis=1)

if pos_dev.argmax() == rot_dev.argmax():
    evict_idx = pos_dev.argmax()
    print('Consensus eviction of index {}'.format(evict_idx))
    sif = poses_minus[evict_idx]
    print('Result : {}'.format(sif))
    np.savetxt(os.path.join(datafolder, 'sensor_in_flange.npytxt'), sif)
