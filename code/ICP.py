import math
from sklearn.neighbors import NearestNeighbors
import numpy as np


def point_based_matching(point_pairs):
    """
    This function is based on the paper "Robot Pose Estimation in Unknown Environments by Matching 2D Range Scans"
    by F. Lu and E. Milios.

    :param point_pairs: [[[x_0, y_0], [x_ref_0, y_ref_0]], [[x_1, y_1], [x_ref_1, y_ref_1]], [[x_180, y_180], [x_ref_180, y_ref_180]]]
    :return: the rotation angle and the 2D translation (x, y) to be applied for matching the given pairs of points
    """

    n = len(point_pairs)
    if n == 0:
        return None, None, None

    x_list = point_pairs[:, 0, 0]
    y_list = point_pairs[:, 0, 1]
    x_ref_list = point_pairs[:, 1, 0]
    y_ref_list = point_pairs[:, 1, 1]

    x_mean = np.mean(x_list)
    y_mean = np.mean(y_list)
    x_ref_mean = np.mean(x_ref_list)
    y_ref_mean = np.mean(y_ref_list)

    x_mean_list = [x_mean for i in range(n)]
    y_mean_list = [y_mean for i in range(n)]
    x_ref_mean_list = [x_ref_mean for i in range(n)]
    y_ref_mean_list = [y_ref_mean for i in range(n)]

    s_x_xref = np.sum((x_list - x_mean_list) * (x_ref_list - x_ref_mean_list))
    s_y_yref = np.sum((y_list - y_mean_list) * (y_ref_list - y_ref_mean_list))
    s_x_yref = np.sum((x_list - x_mean_list) * (y_ref_list - y_ref_mean_list))
    s_y_xref = np.sum((y_list - y_mean_list) * (x_ref_list - x_ref_mean_list))

    rot_angle = math.atan2(s_x_yref - s_y_xref, s_x_xref + s_y_yref)
    translation_x = x_ref_mean - (x_mean * math.cos(rot_angle) - y_mean * math.sin(rot_angle))
    translation_y = y_ref_mean - (x_mean * math.sin(rot_angle) + y_mean * math.cos(rot_angle))

    s = np.sin(rot_angle)
    c = np.cos(rot_angle)
    rot_matrix = np.array([[c, -s], [s, c]])

    return rot_matrix, translation_x, translation_y


def icp_process(points, reference_points, max_iterations=100, distance_threshold=500, convergence_translation_threshold=10,
                convergence_rotation_threshold=1e-3, point_pairs_threshold=10, verbose=False):
    '''
    通过多伦迭代，最终完成icp的匹配过程。
    :param points:
    :param reference_points:
    :param max_iterations:
    :param distance_threshold:
    :param convergence_translation_threshold:
    :param convergence_rotation_threshold:
    :param point_pairs_threshold:
    :param verbose:
    :return:
    '''
    transformation_history = []

    nbrs = NearestNeighbors(n_neighbors=1, algorithm='kd_tree').fit(reference_points) # TODO 目前先使用已有的、基于物理距离的算法来进行point_pair的匹配，之后需要添加语义时，可能要改写一下这里的方法。

    for iter_num in range(max_iterations):
        if verbose:
            print('------ iteration', iter_num, '------')

        closest_point_pairs = []  # list of point correspondences for closest point rule

        distances, indices = nbrs.kneighbors(points)
        for nn_index in range(len(distances)):
            if distances[nn_index][0] < distance_threshold:
                closest_point_pairs.append((points[nn_index], reference_points[indices[nn_index][0]]))

        # if only few point pairs, stop process
        if verbose:
            print('number of pairs found:', len(closest_point_pairs))
        if len(closest_point_pairs) < point_pairs_threshold:
            if verbose:
                print('No better solution can be found (very few point pairs)!')
            break

        # compute translation and rotation using point correspondences
        closest_rot_angle, closest_translation_x, closest_translation_y = point_based_matching(closest_point_pairs)
        if closest_rot_angle is not None:
            if verbose:
                print('Rotation:', math.degrees(closest_rot_angle), 'degrees')
                print('Translation:', closest_translation_x, closest_translation_y)
        if closest_rot_angle is None or closest_translation_x is None or closest_translation_y is None:
            if verbose:
                print('No better solution can be found!')
            break

        # transform 'points' (using the calculated rotation and translation)
        c, s = math.cos(closest_rot_angle), math.sin(closest_rot_angle)
        rot = np.array([[c, -s],
                        [s, c]])
        aligned_points = np.dot(points, rot.T)
        aligned_points[:, 0] += closest_translation_x
        aligned_points[:, 1] += closest_translation_y

        # update 'points' for the next iteration
        points = aligned_points

        # update transformation history
        transformation_history.append(np.hstack((rot, np.array([[closest_translation_x], [closest_translation_y]]))))

        # check convergence
        if (abs(closest_rot_angle) < convergence_rotation_threshold) \
                and (abs(closest_translation_x) < convergence_translation_threshold) \
                and (abs(closest_translation_y) < convergence_translation_threshold):
            if verbose:
                print('Converged!')
            break

    return transformation_history, points
