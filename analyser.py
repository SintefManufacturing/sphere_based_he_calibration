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
ap.add_argument('-o', '--object_spec', type=str,
                help="""File which can be evaluated as Python code, and which specifies
                variables 'obj_dims', giving a pair of object
                dimensions [m] with x-dimension first, and
                'ball_radius', specifying the radius [m] of the object
                balls.""")
ap.add_argument('-i', '--index', nargs='+', type=int,
                help="""Only analyse specified indices. Multiple indices separated by
                spaces may be given.""")
ap.add_argument('-f', '--force_analysis', action='store_true',
                help="""Flag for forced analysis on all, or specified indices, irregardless
                of whether cached analysis results from previous runs
                are found.""")
args = ap.parse_args()

# Set up the data folder and find data files
data_folder = args.data_folder
datafiles = os.listdir(data_folder)

# Read the calibration object specification
if args.object_spec is None or not os.path.isfile(args.object_spec):
    raise Exception('An (existing) "obj_spec" file is required ("{}")'
                    .format(args.object_spec))
obj_dims = None
ball_radius = None
exec(open(args.object_spec).read())
if obj_dims is None or ball_radius is None:
    raise Exception('Either "ball_radius" or "obj_dims" was not specified'
                    + ' in the "obj_spec" file. Was given "{}"'
                    .format(args.object_spec))

# Clear temporary files
[os.remove(pcdf) for pcdf in glob.glob('*.pcd')]

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
