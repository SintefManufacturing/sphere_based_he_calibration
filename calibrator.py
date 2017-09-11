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
import argparse
import glob

import numpy as np
import pcl
import math3d as m3d

from hand_eye_calibration import ParkMartinCalibrator

from naming import scenere, scenetmpl, fibre, fibtmpl, oisre, oistmpl

ap = argparse.ArgumentParser()
ap.add_argument('data_folder', type=str)
args = ap.parse_args()


data_files = os.listdir(args.data_folder)


oindices = []
findices = []
for fn in data_files:
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

fibs = [m3d.Transform(np.loadtxt(os.path.join(args.data_folder, fibtmpl.format(i))))
        for i in indices]
oiss = [m3d.Transform(np.loadtxt(os.path.join(args.data_folder, oistmpl.format(i))))
        for i in indices]
sios = [t.inverse for t in oiss]

fib_sio_pairs = np.array(list(zip(fibs, sios)))
# fib_sio_pairs = np.roll(fib_sio_pairs, 1, axis=0)
np.set_printoptions(precision=3)

pmc = ParkMartinCalibrator(fib_sio_pairs)
# np.savetxt(os.path.join(args.data_folder, 'sensor_in_flange.npytxt'),
#            pmc.sensor_in_flange.pose_vector)
print('Full data result : {}'.format(pmc.sensor_in_flange.pose_vector))

# Simple consensus based eviction by observing the removal of any of
# the pose pairs.

poses_minus = []
for i in range(len(fib_sio_pairs)):
    poses_minus.append(ParkMartinCalibrator(
        np.delete(fib_sio_pairs, i, axis=0)).sensor_in_flange.pose_vector)
poses_minus = np.array(poses_minus)
print(poses_minus)

# Position statistics
pos = poses_minus[:, :3]
pos_avg = np.average(pos, axis=0)
pos_devv = pos - pos_avg
pos_dev = np.linalg.norm(pos_devv, axis=1)
pos_stddev = np.std(pos_dev)

# Rotation statistics
rot = poses_minus[:, 3:]
rot_avg = np.average(rot, axis=0)
rot_devv = rot - rot_avg
rot_dev = np.linalg.norm(rot_devv, axis=1)
rot_stddev = np.std(rot_dev)

# Selection of eviction indices
evict_idx = np.where(np.logical_or((pos_dev-3*pos_stddev) > 0,
                                   (rot_dev-3*rot_stddev) > 0))
print('Consensus eviction of index {}'.format(evict_idx))

pmc_minus = ParkMartinCalibrator(np.delete(fib_sio_pairs, evict_idx, 0))
pmc_minus.sensor_in_flange
print('Consensus result : {}'.format(pmc_minus.sensor_in_flange.pose_vector))
np.savetxt(os.path.join(args.data_folder, 'sensor_in_flange.npytxt'),
           pmc_minus.sensor_in_flange.pose_vector)
