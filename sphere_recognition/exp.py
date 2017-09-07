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

from .euclidean_clustering import EuclideanClusterExtractor
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

def find_object(pc, obj_dims, ball_radius):
    npc = pc.to_array()
    # Crop to cylinder segment
    npc = npc[
        (npc[:, 2] > 0.4) *
        (npc[:, 2] < 1.5) *
        (np.sum(npc[:, :2]**2, axis=1) < 0.5**2)
    ]
    pc=pcl.PointCloud(npc)
    pcl.save(pc,'scene_cropped.pcd')
    print('#PC : {}'.format(pc.size))

    # Downsample
    vg = pc.make_voxel_grid_filter()
    vg.set_leaf_size(*(3*[0.005]))
    pc = vg.filter()
    npc=pc.to_array()
    pcl.save(pc,'scene_downsampled.pcd')
    print('#PC : {}'.format(pc.size))

    # Remove extension of planes of considerable goodness
    pi = 0
    print('\n*****\nFiltering planes\n*****')
    while True:
        print('#PC : {}'.format(pc.size))
        psegm = pc.make_segmenter_normals(ksearch=5)
        psegm.set_model_type(pcl.SACMODEL_PLANE)
        psegm.set_normal_distance_weight(0.1)
        psegm.set_optimize_coefficients(True)
        psegm.set_method_type(pcl.SAC_RANSAC)
        psegm.set_distance_threshold(0.001)
        psegm.set_max_iterations(1e4)
        pidx, pmodel = psegm.segment()
        print('Identified plane: #{} {}'.format(len(pidx), pmodel))
        if len(pidx) < 1000:
            break
        # Match points with more tolerance. Using inner product with the
        # plane vector, we may find the signed distance to the plane in
        # the cloud.
        npvec = m3d.geometry.Plane(coeffs=pmodel).plane_vector
        pdists = np.abs(npvec.array.dot(pc.to_array().T) - 1)
        # Find indexes of the extended plane
        xpidx = np.where(pdists < 0.005)[0]
        print('Removing extended plane points: #{} {}'.format(len(xpidx), pmodel))
        pl = pc.extract(xpidx)
        pcl.save(pl, 'plane_{}.pcd'.format(pi))
        pc = pc.extract(xpidx, negative=True)
        pi += 1
    npc = pc.to_array()
    pcl.save(pc,'scene_curved.pcd')
    print('#PC : {}'.format(pc.size))


    # Remove isolated points
    print('Filtering outliers')
    sor = pc.make_statistical_outlier_filter()
    sor.set_mean_k(100)
    sor.set_std_dev_mul_thresh(1)
    pc = sor.filter()
    print('#PC: {}'.format(pc.size))
    npc=pc.to_array()
    pcl.save(pc,'scene_dense.pcd')
    print('#PC : {}'.format(pc.size))


    print('\n*****\nFinding spheres\n*****')
    def find_spheres(pc, radius, n_max=1):
        spheres = []
        si = 0
        idx = [None]
        while len(idx) > 0 and pc.size > 0 and si < n_max:
            segm = pc.make_segmenter_normals(ksearch=10)
            segm.set_model_type(pcl.SACMODEL_NORMAL_SPHERE)
            segm.set_normal_distance_weight(0.001)
            segm.set_optimize_coefficients(True)
            segm.set_distance_threshold(0.001)
            segm.set_radius_limits(0.95*radius, 1.05*radius)
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
                pcl.save(pc, 'scene_after_{}.pcd'.format(si))
            else:
                break
            si += 1
        return spheres
    # Finding three spheres directly in scene:
    #spheres = find_spheres(pc, n_max=3)
    # Finding spheres based on Euclidean Clustering
    ece = EuclideanClusterExtractor(nn_dist=0.005, min_pts=50,
                                    min_max_length=0.7*ball_radius,
                                    max_max_length=2.5*ball_radius)
    ecs = ece.extract(pc)
    spheres = []
    for ec in ecs:
        ec_sphere = find_spheres(ec, ball_radius, n_max=1)
        spheres += ec_sphere
    n_sph = len(spheres)
    for i in range(n_sph):
        pcl.save(spheres[i][0], 'sphere_{}.pcd'.format(i))
    spheres.sort(key=lambda s:s[0].size, reverse=True)
    centres = [m3d.Vector(sph[1][:3]) for sph in spheres[:3]]
    n_centres = len(centres)

    # Setup distance matrix
    distm = spsp.distance.squareform(spsp.distance.pdist([c.array for c in centres]))
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
        print('Match was ambiguous (#matches={})'.format(len(matchidxs)))
        return None
    if len(matchidxs) == 0:
        print('No match found.')
        return None
    # Identify indexes for sphere 0
    sph_0_idx = matchidxs[0]
    # Sphere x and y indexes where distances match
    sph_x_idx = np.abs(distm[sph_0_idx]-obj_dims[0]).argmin()
    sph_y_idx = np.abs(distm[sph_0_idx]-obj_dims[1]).argmin()
    print('Matching indices: 0:{}, x:{}, y:{}'.format(sph_0_idx,sph_x_idx,sph_y_idx))

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

