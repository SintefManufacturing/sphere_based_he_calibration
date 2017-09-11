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

from sphere_recognition import find_object
from naming import scenere, scenetmpl, fibre, fibtmpl, oisre, oistmpl


ap = argparse.ArgumentParser()
ap.add_argument('data_folder', type=str)
ap.add_argument('--indices', nargs='+', type=int,
                help="""Only analyse specified indices""")
ap.add_argument('--force_analysis', action='store_true')
args = ap.parse_args()

# data_folder = 'data_' + datetime.date.today().isoformat()
data_folder = args.data_folder
datafiles = os.listdir(data_folder)

# Clear temporary files
[os.remove(pcdf) for pcdf in glob.glob('*.pcd')]

obj_dims = (0.15, 0.12)
ball_radius = 0.037

sindices = set()
findices = set()
for fn in datafiles:
    sm = scenere.match(fn)
    if sm is not None:
        sindices.add(int(sm.groups()[0]))
        continue
    fm = fibre.match(fn)
    if fm is not None:
        findices.add(int(fm.groups()[0]))
print(findices, sindices)
if findices != sindices:
    print('!!! WARNING !!! Mismatch between fib pose ({}) and scene ({}) indices'
          .format(sorted(findices), sorted(sindices)))

if args.indices is not None:
    indices = sindices.intersection(findices).intersection(args.indices)
else:
    indices = sindices.intersection(findices)

# indices.sort()

scenes = []
fibs = []
oiss = []
valid_indices = []
for i in indices:
    oisfname = oistmpl.format(i)
    if oisfname in datafiles and not args.force_analysis:
        print('Object transform exists for index {}, skipping.'.format(i))
        valid_indices.append(i)
        continue
    print('Analysing index {}'.format(i))
    scene = pcl.load(os.path.join(data_folder, scenetmpl.format(i)))
    ois = find_object(scene, obj_dims, ball_radius)
    if ois is not None:
        np.savetxt(os.path.join(data_folder, oisfname), ois.pose_vector)
        scenes.append(scene)
        oiss.append(ois)
        fibs.append(np.loadtxt(os.path.join(data_folder, fibtmpl.format(i))))
        valid_indices.append(i)
