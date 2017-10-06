# coding=utf-8

"""
"""

__author__ = "Morten Lind"
__copyright__ = "Morten Lind, SINTEF 2017"
__credits__ = ["Morten Lind"]
__license__ = "GPLv3"
__maintainer__ = "Morten Lind"
__email__ = "morten@lind.dyndns.dk, morten.lind@sintef.no"
__status__ = "Development"

import os
import datetime
import argparse

import numpy as np
import pcl

from urx import Robot
from pc_grabber import PCG

from naming import scenetmpl, fibtmpl

ap = argparse.ArgumentParser()
ap.add_argument('-d', '--data_folder', type=str,
                default='data_' + datetime.datetime.now().isoformat()[:19])
ap.add_argument('robot_host', type=str)
# Lab-trondheim '192.168.0.90'
# Lab Giørtz '10.0.0.100'

args = ap.parse_args()

os.makedirs(args.data_folder, exist_ok=True)

pcg = PCG()
rob = Robot(args.robot_host)
rob.set_tcp(6*[0])
rob.set_payload(2.0)

i = 0
while True:
    key = input('Pose robot for sample {}, press [Ret] when ready. Press "e" and [Ret] for stopping'.format(i))
    if key == 'e':
        break
    p = rob.get_pose().pose_vector
    np.savetxt(os.path.join(args.data_folder, fibtmpl.format(i)), p)
    npc = pcg.get_pc()
    pc = pcl.PointCloud(npc)
    pcl.save(pc, os.path.join(args.data_folder, scenetmpl.format(i)))
    i += 1

rob.close()
