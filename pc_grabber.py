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

import cyni
import numpy as np

class PCG:
    def __init__(self):
        cyni.initialize()
        self.device = cyni.getAnyDevice()
        self.device.open()
        self.depthStream = self.device.createStream(b"depth", fps=30)
        self.depthStream.start()

    def __del__(self):
        self.device.close()

    def get_pc(self, save=True, as_array=False):
        """
        Gets pointcloud scene, coords referred to 3D sensor coord. system

        Returns:
            cloud(numpy array): Pointcloud
        """
        depthFrame = self.depthStream.readFrame()
        cloud = cyni.depthMapToPointCloud(depthFrame.data, self.depthStream)
        # cloud *= np.array(cfg.sensor_scale_factor, dtype=np.float32)
        cloud /= 1000.0
        cloud.shape = (cloud.shape[0] * cloud.shape[1], 3)
        # cloud *= [1.02, 1.015, 1.0]
        # cloud *= 1.01
        return cloud.astype(np.float32)
