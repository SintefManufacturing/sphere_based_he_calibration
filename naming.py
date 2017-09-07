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

import re

scenere = re.compile(r'scene_([0-9]{3})\.pcd')
scenetmpl = 'scene_{:03d}.pcd'
fibre = re.compile(r'fib_([0-9]{3})\.npytxt')
fibtmpl = 'fib_{:03d}.npytxt'
oisre = re.compile(r'ois_([0-9]{3})\.npytxt')
oistmpl = 'ois_{:03d}.npytxt'
