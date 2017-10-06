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

import logging
logging.basicConfig(level=logging.DEBUG)
import math3d as m3d
import math3d.geometry
import pcl
import numpy as np
import scipy.spatial as spsp

from srm_pc_utils.plane_segmentation import PlaneSegmenter
from srm_pc_utils.euclidean_clustering import EuclideanClusterExtractor
logging.getLogger('ECE').setLevel(logging.INFO)

# #live = False
# live = True
# if live:
#     pcg = PCG()
#     npc = pcg.get_pc()
#     pc=pcl.PointCloud(npc)
#     pcl.save(pc,'scene.pcd')
# else:
#     pc = pcl.load('scene.pcd')
#     npc = pc.to_array()
#     print('!!!!! Warning !!!!! Using stored scene data !')


def find_spheres(pc, radius, n_max=1):
    spheres = []
    si = 0
    idx = [None]
    while len(idx) > 0 and pc.size > 0 and si < n_max:
        segm = pc.make_segmenter_normals(ksearch=5)
        segm.set_model_type(pcl.SACMODEL_NORMAL_SPHERE)
        segm.set_normal_distance_weight(0.001)
        segm.set_optimize_coefficients(True)
        segm.set_distance_threshold(0.001)
        segm.set_radius_limits(0.9*radius, 1.1*radius)
        segm.set_max_iterations(1e7)
        idx, model = segm.segment()
        if len(idx) == 0:
            break
        print('sphere identified: {} {}'.format(model[-1], len(idx)))
        sphere = pc.extract(idx)
        spheres.append((sphere, model))
        # Filter out region using a kd-tree. Is it cheaper to calculate explicit distance to centre?
        k = spsp.cKDTree(pc.to_array())
        kill_idx = k.query_ball_point(model[:3], 1.25 * model[3])
        pc=pc.extract(kill_idx, negative=True)
        if pc.size > 0:
            pcl.save(pc, 'tmp/scene_after_{}.pcd'.format(si))
        else:
            break
        si += 1
    return spheres


def find_object(pc, obj_dims, ball_radius):
    npc = pc.to_array()
    # Crop to cylinder segment
    print('Finding object (#PC {})'.format(pc.size))
    npc = npc[
        (npc[:, 2] > 0.4)
        * (npc[:, 2] < 1.5)
        # * (np.sum(npc[:, :2]**2, axis=1) < 0.5**2)
    ]
    pc = pcl.PointCloud(npc)
    pcl.save(pc, 'tmp/scene_cropped.pcd')
    print('Cropped (#PC {})'.format(pc.size))

    # Downsample
    point_distance = 0.01
    face_density = 1 / point_distance**2 
    vg = pc.make_voxel_grid_filter()
    vg.set_leaf_size(*(3*[point_distance]))
    pc = vg.filter()
    npc = pc.to_array()
    pcl.save(pc, 'tmp/scene_downsampled.pcd')
    print('Downsampled (#PC {})'.format(pc.size))

    # Remove extension of planes of considerable goodness
    psegm = PlaneSegmenter(distance_tolerance=0.001,
                           normal_distance_weight=0.1,
                           consume_distance=0.005,
                           maximum_iterations=1e4,
                           minimum_plane_points=500,
                           minimum_density=0.3*face_density)
    planes, pc = psegm(pc)
    npc = pc.to_array()
    pcl.save(pc, 'tmp/scene_curved.pcd')
    print('Filtered planes (#PC {})'.format(pc.size))

    # Remove isolated points
    sor = pc.make_statistical_outlier_filter()
    sor.set_mean_k(10)
    sor.set_std_dev_mul_thresh(0.1)
    pc = sor.filter()
    print('#PC: {}'.format(pc.size))
    npc = pc.to_array()
    pcl.save(pc, 'tmp/scene_dense.pcd')
    print('Filtered outliers (#PC {})'.format(pc.size))

    # Finding three spheres directly in scene:
    # spheres = find_spheres(pc, n_max=3)
    # Finding spheres based on Euclidean Clustering
    ece = EuclideanClusterExtractor(nn_dist=2*point_distance,  # ball_radius*(1-np.cos(np.pi/6)),
                                    min_pts=10,
                                    min_max_length=0.3*ball_radius,
                                    max_max_length=2.3*ball_radius)
    ecs = ece.extract(pc)
    print('Extracted {} clusters'.format(len(ecs)))
    if len(ecs) == 0:
        print('!!!!!!!\n!!!!!! No Clusters found !!!!!!\n!!!!!!!\n')
        return None
    spheres = []
    for i, ec in enumerate(ecs):
        pcl.save(ec, 'tmp/cluster_{:03d}.pcd'.format(i))
        ec_spheres = find_spheres(ec, ball_radius, n_max=1)
        spheres += ec_spheres
    spheres = [sph for sph in spheres if sph[0].size > 10]
    n_sph = len(spheres)
    spheres.sort(key=lambda s: s[0].size, reverse=True)
    for i in range(n_sph):
        pcl.save(spheres[i][0], 'tmp/sphere_{}.pcd'.format(i))
    centres = [m3d.Vector(sph[1][:3]) for sph in spheres]
    n_centres = len(centres)

    # Setup distance matrix
    distm = spsp.distance.squareform(
        spsp.distance.pdist([c.array for c in centres]))
    # distm = np.zeros((n_centres,n_centres))
    # for i in range(n_centres):
    #     for j in range(n_centres):
    #         distm[i,j] = (centres[i] - centres[j]).length
    print(distm)

    # Match matrix
    matchm0 = np.abs(distm-obj_dims[0]) < 0.01
    matchm1 = np.abs(distm-obj_dims[1]) < 0.01
    match = np.logical_and(matchm0.any(axis=1), matchm1.any(axis=1))

    # Test if there is unambiguous object match
    matchidxs = np.where(match)[0]
    if len(matchidxs) > 1:
        print('!!!!!! Match was ambiguous (#matches={}) !!!!!!\n'
              .format(len(matchidxs)))
        return None
    if len(matchidxs) == 0:
        print('!!!!!!!\n!!!!!! No match found. !!!!!!\n!!!!!!!\n')
        return None
    # Identify indexes for sphere 0
    sph_0_idx = matchidxs[0]
    # Sphere x and y indexes where distances match
    errs_x = np.abs(distm[sph_0_idx]-obj_dims[0])
    errs_y = np.abs(distm[sph_0_idx]-obj_dims[1])
    sph_x_idx = errs_x.argmin()
    sph_y_idx = errs_y.argmin()
    err_x = errs_x[sph_x_idx]
    err_y = errs_y[sph_y_idx]
    print('Matching indices: 0:{}, x:{} ({}), y:{} ({})'
          .format(sph_0_idx, sph_x_idx, err_x, sph_y_idx, err_y))

    # Single out the centres and form the x- and y-unit vectors.
    sph_0 = centres[sph_0_idx]
    sph_x = centres[sph_x_idx]
    sph_y = centres[sph_y_idx]
    d_x = (sph_x-sph_0).normalized
    d_y = (sph_y-sph_0).normalized

    # Form the object transform
    t_obj = m3d.Transform.new_from_xyp(d_x, d_y, sph_0)
    return t_obj

    # cdiffs = np.roll(centres,1) - centres
    # ncdiffs = [d.length for d in cdiffs]
    # radii =  [sph[1][3] for sph in spheres]
    # npoints =  [sph[0] for sph in spheres]
