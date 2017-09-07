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

import numpy as np
import pcl

from urx import Robot
from pc_grabber import PCG

from naming import scenetmpl, fibtmpl

datafolder = 'data_' + datetime.date.today().isoformat()
os.makedirs(datafolder)

pcg = PCG()
rob = Robot('192.168.0.90')
rob.set_tcp(6*[0])
rob.set_payload(3.0)

i = 0
while True:
    input('Pose robot for sample {}, press [Ret] when ready'.format(i))
    p = rob.get_pose().pose_vector
    np.savetxt(os.path.join(datafolder, fibtmpl.format(i)), p)
    npc = pcg.get_pc()
    pc = pcl.PointCloud(npc)
    pcl.save(pc, os.path.join(datafolder, scenetmpl.format(i)))
    i += 1
