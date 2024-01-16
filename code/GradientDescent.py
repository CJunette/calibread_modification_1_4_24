import numpy as np
import torch

import UtilFunctions
import configs


def compute_partial_derivative_of_matrix(matrix, source_points, target_points):
    '''
    计算矩阵的偏导数。
    :param matrix:
    :param source_points:
    :param target_points:
    :return:
    '''
    partial_derivative_matrix = np.zeros(matrix.shape)
    for point_index in range(len(source_points)):
        source_point = source_points[point_index]
        target_point = target_points[point_index]
        sum_1 = 2 * np.outer(np.dot(matrix, source_point), source_point)
        sum_2 = 2 * np.outer(target_point, source_point)
        partial_derivative_matrix += sum_1 - sum_2

    return partial_derivative_matrix


def compute_Frobenius_inner_product(matrix_1, matrix_2):
    '''
    计算矩阵的Frobenius内积。
    :param matrix_1:
    :param matrix_2:
    :return:
    '''
    inner_product = np.sum(np.multiply(matrix_1, matrix_2))
    return inner_product


def gradient_descent_with_rotation_and_translation(point_pairs, learning_rate_theta=1e-12, learning_rate_x=5e-5, learning_rate_y=5e-5, num_iterations=500):
    theta = 0
    t_x = 0
    t_y = 0

    source_points = np.array([UtilFunctions.change_2d_vector_to_homogeneous_vector(point_pair[0]) for point_pair in point_pairs])
    target_points = np.array([UtilFunctions.change_2d_vector_to_homogeneous_vector(point_pair[1]) for point_pair in point_pairs])

    transform_matrix = np.array([[np.cos(theta), -np.sin(theta), t_x],
                                 [np.sin(theta), np.cos(theta), t_y],
                                 [0, 0, 1]])

    for i in range(num_iterations):
        # 这里在公式推导上肯定还有点问题，theta在每次更新时显然都应该是一个新的值，而不是一个3*3的矩阵。
        # 这里公式推导虽然有点问题，但已经确定了常见的解法。
        partial_derivative_matrix = compute_partial_derivative_of_matrix(transform_matrix, source_points, target_points)
        new_theta = theta - learning_rate_theta * compute_Frobenius_inner_product(partial_derivative_matrix, np.array([[-np.sin(theta), -np.cos(theta), 0],
                                                                                                                       [np.cos(theta), -np.sin(theta), 0],
                                                                                                                       [0, 0, 0]]))
        new_t_x = t_x - learning_rate_x * compute_Frobenius_inner_product(partial_derivative_matrix, np.array([[0, 0, 1],
                                                                                                               [0, 0, 0],
                                                                                                               [0, 0, 0]]))
        new_t_y = t_y - learning_rate_y * compute_Frobenius_inner_product(partial_derivative_matrix, np.array([[0, 0, 0],
                                                                                                               [0, 0, 1],
                                                                                                               [0, 0, 0]]))

        transform_matrix = np.array([[np.cos(new_theta), -np.sin(new_theta), new_t_x],
                                     [np.sin(new_theta), np.cos(new_theta), new_t_y],
                                     [0, 0, 1]])

        # if abs(new_theta - theta) < 1e-2 and abs(new_t_x - t_x) < 1 and abs(new_t_y - t_y) < 1:
        #     print(f"converged at iteration: {i}")
        #     break

        source_points_after_transform = np.array([np.dot(transform_matrix, source_point) for source_point in source_points])
        error = np.sum(np.square(source_points_after_transform - target_points))
        print(f"iteration: {i}, error: {error}")

        theta = new_theta
        t_x = new_t_x
        t_y = new_t_y

    return transform_matrix


def gradient_descent_with_whole_matrix(point_pairs,
                                       learning_rate_00=1e-9, learning_rate_01=1e-9, learning_rate_02=5e-4,
                                       learning_rate_10=1e-9, learning_rate_11=1e-9, learning_rate_12=5e-4,
                                       learning_rate_20=1e-9, learning_rate_21=1e-9, learning_rate_22=1e-8, num_iterations=1000):
    # TODO 这里的learning_rate有可能需要每个参数单独设置。
    source_points = np.array([UtilFunctions.change_2d_vector_to_homogeneous_vector(point_pair[0]) for point_pair in point_pairs])
    target_points = np.array([UtilFunctions.change_2d_vector_to_homogeneous_vector(point_pair[1]) for point_pair in point_pairs])

    matrix_00 = 1 + 1e-8
    matrix_01 = 1e-8
    matrix_02 = 1e-8
    matrix_10 = 1e-8
    matrix_11 = 1 + 1e-8
    matrix_12 = 1e-8
    matrix_20 = 1e-8
    matrix_21 = 1e-8
    matrix_22 = 1 + 1e-8

    transform_matrix = np.array([[matrix_00, matrix_01, matrix_02],
                                 [matrix_10, matrix_11, matrix_12],
                                 [matrix_20, matrix_21, matrix_22]])

    for i in range(num_iterations):
        partial_derivative_matrix = compute_partial_derivative_of_matrix(transform_matrix, source_points, target_points)

        new_matrix_00 = matrix_00 - learning_rate_00 * compute_Frobenius_inner_product(partial_derivative_matrix, np.array([[1, 0, 0],
                                                                                                                            [0, 0, 0],
                                                                                                                            [0, 0, 0]]))
        new_matrix_01 = matrix_01 - learning_rate_01 * compute_Frobenius_inner_product(partial_derivative_matrix, np.array([[0, 1, 0],
                                                                                                                            [0, 0, 0],
                                                                                                                            [0, 0, 0]]))
        new_matrix_02 = matrix_02 - learning_rate_02 * compute_Frobenius_inner_product(partial_derivative_matrix, np.array([[0, 0, 1],
                                                                                                                            [0, 0, 0],
                                                                                                                            [0, 0, 0]]))
        new_matrix_10 = matrix_10 - learning_rate_10 * compute_Frobenius_inner_product(partial_derivative_matrix, np.array([[0, 0, 0],
                                                                                                                            [1, 0, 0],
                                                                                                                            [0, 0, 0]]))
        new_matrix_11 = matrix_11 - learning_rate_11 * compute_Frobenius_inner_product(partial_derivative_matrix, np.array([[0, 0, 0],
                                                                                                                            [0, 1, 0],
                                                                                                                            [0, 0, 0]]))
        new_matrix_12 = matrix_12 - learning_rate_12 * compute_Frobenius_inner_product(partial_derivative_matrix, np.array([[0, 0, 0],
                                                                                                                            [0, 0, 1],
                                                                                                                            [0, 0, 0]]))
        new_matrix_20 = matrix_20 - learning_rate_20 * compute_Frobenius_inner_product(partial_derivative_matrix, np.array([[0, 0, 0],
                                                                                                                            [0, 0, 0],
                                                                                                                            [1, 0, 0]]))
        new_matrix_21 = matrix_21 - learning_rate_21 * compute_Frobenius_inner_product(partial_derivative_matrix, np.array([[0, 0, 0],
                                                                                                                            [0, 0, 0],
                                                                                                                            [0, 1, 0]]))
        new_matrix_22 = matrix_22 - learning_rate_22 * compute_Frobenius_inner_product(partial_derivative_matrix, np.array([[0, 0, 0],
                                                                                                                            [0, 0, 0],
                                                                                                                            [0, 0, 1]]))

        transform_matrix = np.array([[new_matrix_00, new_matrix_01, new_matrix_02],
                                     [new_matrix_10, new_matrix_11, new_matrix_12],
                                     [new_matrix_20, new_matrix_21, new_matrix_22]])

        # if abs(new_theta - theta) < 1e-2 and abs(new_t_x - t_x) < 1 and abs(new_t_y - t_y) < 1:
        #     print(f"converged at iteration: {i}")
        #     break

        source_points_after_transform = np.array([np.dot(transform_matrix, source_point) for source_point in source_points])
        error = np.sum(np.square(source_points_after_transform - target_points))
        # print(f"iteration: {i}, error: {error}")
        matrix_00 = new_matrix_00
        matrix_01 = new_matrix_01
        matrix_02 = new_matrix_02
        matrix_10 = new_matrix_10
        matrix_11 = new_matrix_11
        matrix_12 = new_matrix_12
        matrix_20 = new_matrix_20
        matrix_21 = new_matrix_21
        matrix_22 = new_matrix_22

    return transform_matrix


def gradient_descent_with_whole_matrix_using_tensor(point_pairs,
                                                    learning_rate_00=1e-7, learning_rate_01=1e-7, learning_rate_02=5e-1,
                                                    learning_rate_10=1e-7, learning_rate_11=1e-7, learning_rate_12=5e-1,
                                                    learning_rate_20=1e-7, learning_rate_21=1e-7, learning_rate_22=1e-7, max_iterations=1000):
    '''
    与之前gradient_descent_with_whole_matrix()相比，这里的差别在于使用了tensor加速，且用的是tensor自己的反向传播及求导。
    :param point_pairs: [[[source_point_x_0, source_point_y_0], [source_point_x_1, source_point_y_1], ...], [[target_point_x_0, target_point_y_0], [target_point_x_1, target_point_y_1], ...]]
    :param learning_rate_00:
    :param learning_rate_01:
    :param learning_rate_02:
    :param learning_rate_10:
    :param learning_rate_11:
    :param learning_rate_12:
    :param learning_rate_20:
    :param learning_rate_21:
    :param learning_rate_22:
    :param max_iterations:
    :return:
    '''
    source_points = np.array([UtilFunctions.change_2d_vector_to_homogeneous_vector(point_pair[0]) for point_pair in point_pairs])
    target_points = np.array([UtilFunctions.change_2d_vector_to_homogeneous_vector(point_pair[1]) for point_pair in point_pairs])

    source_points = torch.tensor(source_points, dtype=torch.float32, requires_grad=False).cuda(0)
    target_points = torch.tensor(target_points, dtype=torch.float32, requires_grad=False).cuda(0)

    matrix_00 = 1 + 1e-5
    matrix_01 = 1e-5
    matrix_02 = 1e-5
    matrix_10 = 1e-5
    matrix_11 = 1 + 1e-5
    matrix_12 = 1e-5
    matrix_20 = 1e-5
    matrix_21 = 1e-5
    matrix_22 = 1 + 1e-5

    transform_matrix = torch.tensor([[matrix_00, matrix_01, matrix_02],
                                     [matrix_10, matrix_11, matrix_12],
                                     [matrix_20, matrix_21, matrix_22]], dtype=torch.float32, requires_grad=True).cuda(0)

    for i in range(max_iterations):
        transform_matrix.retain_grad()  # 保证transform_matrix的梯度能够一直被保留。
        source_points_after_transform = torch.matmul(transform_matrix, source_points.transpose(0, 1)).transpose(0, 1)
        error = torch.mean(torch.square(source_points_after_transform - target_points))
        error.backward()
        # print(transform_matrix.grad)

        with torch.no_grad():
            transform_matrix[0][0] -= learning_rate_00 * transform_matrix.grad[0][0]
            transform_matrix[0][1] -= learning_rate_01 * transform_matrix.grad[0][1]
            transform_matrix[0][2] -= learning_rate_02 * transform_matrix.grad[0][2]
            transform_matrix[1][0] -= learning_rate_10 * transform_matrix.grad[1][0]
            transform_matrix[1][1] -= learning_rate_11 * transform_matrix.grad[1][1]
            transform_matrix[1][2] -= learning_rate_12 * transform_matrix.grad[1][2]
            transform_matrix[2][0] -= learning_rate_20 * transform_matrix.grad[2][0]
            transform_matrix[2][1] -= learning_rate_21 * transform_matrix.grad[2][1]
            transform_matrix[2][2] -= learning_rate_22 * transform_matrix.grad[2][2]

        transform_matrix.grad.zero_()
        print(f"iteration: {i}, error: {error}")

    transform_matrix = transform_matrix.cpu().detach().numpy()
    return transform_matrix


def gradient_descent_with_whole_matrix_using_tensor_with_weight(point_pairs, weight, last_iteration_num,
                                                                learning_rate_00=1e-7, learning_rate_01=1e-7, learning_rate_02=1,
                                                                learning_rate_10=1e-7, learning_rate_11=1e-7, learning_rate_12=1,
                                                                learning_rate_20=1e-7, learning_rate_21=1e-7, learning_rate_22=1e-6, max_iterations=2000):
    '''
    与之前gradient_descent_with_whole_matrix()相比，这里的差别在于使用了tensor加速，且用的是tensor自己的反向传播及求导。
    :param weight: 根据文本点的特征和对应注视点的特征，设计的权重向量。
    :param point_pairs: [[[source_point_x_0, source_point_y_0], [source_point_x_1, source_point_y_1], ...], [[target_point_x_0, target_point_y_0], [target_point_x_1, target_point_y_1], ...]]
    :param learning_rate_00:
    :param learning_rate_01:
    :param learning_rate_02:
    :param learning_rate_10:
    :param learning_rate_11:
    :param learning_rate_12:
    :param learning_rate_20:
    :param learning_rate_21:
    :param learning_rate_22:
    :param max_iterations:
    :return:
    '''
    stop_accuracy = configs.gradient_descent_stop_accuracy

    if 10 < last_iteration_num < 200:
        learning_rate_00 *= 1
        learning_rate_01 *= 1
        learning_rate_02 *= 1
        learning_rate_10 *= 1
        learning_rate_11 *= 1
        learning_rate_12 *= 1
        learning_rate_20 *= 1
        learning_rate_21 *= 1
        learning_rate_22 *= 1
        stop_accuracy *= 10

    source_points = np.array([UtilFunctions.change_2d_vector_to_homogeneous_vector(point_pair[0]) for point_pair in point_pairs])
    target_points = np.array([UtilFunctions.change_2d_vector_to_homogeneous_vector(point_pair[1]) for point_pair in point_pairs])
    source_points = torch.tensor(source_points, dtype=torch.float32, requires_grad=False).cuda(0)
    target_points = torch.tensor(target_points, dtype=torch.float32, requires_grad=False).cuda(0)

    weight_tensor = torch.tensor(weight, dtype=torch.float32, requires_grad=False).unsqueeze(-1).cuda(0)

    matrix_00 = 1 + 1e-5
    matrix_01 = 1e-5
    matrix_02 = 1e-5
    matrix_10 = 1e-5
    matrix_11 = 1 + 1e-5
    matrix_12 = 1e-5
    matrix_20 = 1e-5
    matrix_21 = 1e-5
    matrix_22 = 1 + 1e-5

    transform_matrix = torch.tensor([[matrix_00, matrix_01, matrix_02],
                                     [matrix_10, matrix_11, matrix_12],
                                     [matrix_20, matrix_21, matrix_22]], dtype=torch.float32, requires_grad=True).cuda(0)
    last_error = 1000000
    last_iteration_num = 0
    for i in range(max_iterations):
        transform_matrix.retain_grad()  # 保证transform_matrix的梯度能够一直被保留。
        source_points_after_transform = torch.matmul(transform_matrix, source_points.transpose(0, 1)).transpose(0, 1)
        square = torch.square(source_points_after_transform - target_points)
        square_with_weight = square * weight_tensor
        error = torch.mean(square_with_weight)
        error.backward()
        # print(f"grad length: {torch.norm(transform_matrix.grad)}")

        with (torch.no_grad()):
            # if i < 50 and \
            #     abs(transform_matrix.grad[0][0]) < 1e3 and \
            #     abs(transform_matrix.grad[0][1]) < 1e3 and \
            #     abs(transform_matrix.grad[0][2]) < 1e-1 and \
            #     abs(transform_matrix.grad[1][0]) < 1e3 and \
            #     abs(transform_matrix.grad[1][1]) < 1e3 and \
            #     abs(transform_matrix.grad[1][2]) < 1e-1 and \
            #     abs(transform_matrix.grad[2][0]) < 1e-1 and \
            #     abs(transform_matrix.grad[2][1]) < 1e-1 and \
            #     abs(transform_matrix.grad[2][2]) < 1e-3:
            #     learning_rate_00 *= 100
            #     learning_rate_01 *= 100
            #     learning_rate_02 *= 100
            #     learning_rate_10 *= 100
            #     learning_rate_11 *= 100
            #     learning_rate_12 *= 100
            #     learning_rate_20 *= 100
            #     learning_rate_21 *= 100
            #     learning_rate_22 *= 100

            transform_matrix[0][0] -= learning_rate_00 * transform_matrix.grad[0][0]
            transform_matrix[0][1] -= learning_rate_01 * transform_matrix.grad[0][1]
            transform_matrix[0][2] -= learning_rate_02 * transform_matrix.grad[0][2]
            transform_matrix[1][0] -= learning_rate_10 * transform_matrix.grad[1][0]
            transform_matrix[1][1] -= learning_rate_11 * transform_matrix.grad[1][1]
            transform_matrix[1][2] -= learning_rate_12 * transform_matrix.grad[1][2]
            transform_matrix[2][0] -= learning_rate_20 * transform_matrix.grad[2][0]
            transform_matrix[2][1] -= learning_rate_21 * transform_matrix.grad[2][1]
            transform_matrix[2][2] -= learning_rate_22 * transform_matrix.grad[2][2]

        # print(transform_matrix.grad, f"iteration: {i}, error: {error}")
        transform_matrix.grad.zero_()
        print(f"iteration: {i}, error: {error}")
        last_iteration_num = i
        if abs(last_error - error) < stop_accuracy:
            break
        else:
            last_error = error

    transform_matrix = transform_matrix.cpu().detach().numpy()
    return transform_matrix, last_error, last_iteration_num


def gradient_descent_with_torch(point_pairs, weight, last_iteration_num, learning_rate=2e-1, max_iterations=2000, stop_grad_norm=5):
    torch.manual_seed(configs.random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    source_points = np.array([UtilFunctions.change_2d_vector_to_homogeneous_vector(point_pair[0]) for point_pair in point_pairs])
    target_points = np.array([UtilFunctions.change_2d_vector_to_homogeneous_vector(point_pair[1]) for point_pair in point_pairs])
    source_points = torch.tensor(source_points, dtype=torch.float32, requires_grad=False).cuda(0)
    target_points = torch.tensor(target_points, dtype=torch.float32, requires_grad=False).cuda(0)

    weight_tensor = torch.tensor(weight, dtype=torch.float32, requires_grad=False).unsqueeze(-1).cuda(0)

    matrix_00 = 1 + 1e-5
    matrix_01 = 1e-5
    matrix_02 = 1e-5
    matrix_10 = 1e-5
    matrix_11 = 1 + 1e-5
    matrix_12 = 1e-5
    matrix_20 = 1e-5
    matrix_21 = 1e-5
    matrix_22 = 1 + 1e-5

    transform_matrix = torch.tensor([[matrix_00, matrix_01, matrix_02],
                                     [matrix_10, matrix_11, matrix_12],
                                     [matrix_20, matrix_21, matrix_22]], dtype=torch.float32, requires_grad=True).cuda(0)
    transform_matrix = torch.nn.Parameter(transform_matrix)  # 将transform_matrix转换为一个可以优化的参数

    optimizer = torch.optim.Adam([transform_matrix], lr=learning_rate)
    last_error = 1000000
    grad_norm_list = []
    square_scale_tensor = torch.tensor([1, 1, 1e4], dtype=torch.float32, requires_grad=False).cuda(0)

    for iteration_index in range(max_iterations):
        optimizer.zero_grad()
        source_points_after_transform = torch.matmul(transform_matrix, source_points.transpose(0, 1)).transpose(0, 1)
        square = torch.square(source_points_after_transform - target_points)
        square = square * square_scale_tensor
        square_with_weight = square * weight_tensor
        error = torch.mean(square_with_weight)
        error.backward()
        optimizer.step()

        # print(f"iteration: {iteration_index}, error: {error}, grad_norm: {torch.norm(transform_matrix.grad)}")
        last_iteration_num = iteration_index
        last_error = error
        grad_norm_list.append(torch.norm(transform_matrix.grad))
        bool_stop = True
        for grad_norm in grad_norm_list[-50:]:
            if grad_norm > stop_grad_norm:
                bool_stop = False
                break

        if last_iteration_num > 1000 and bool_stop:
            last_error = error
            break
        else:
            last_error = error

    transform_matrix = transform_matrix.cpu().detach().numpy()
    return transform_matrix, last_error, last_iteration_num


def gradient_descent_with_torch_and_batch(point_pairs, weight, last_iteration_num, learning_rate=1e-2, max_iterations=2000):
    torch.manual_seed(configs.random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    source_points = np.array([UtilFunctions.change_2d_vector_to_homogeneous_vector(point_pair[0]) for point_pair in point_pairs])
    target_points = np.array([UtilFunctions.change_2d_vector_to_homogeneous_vector(point_pair[1]) for point_pair in point_pairs])
    # target_points = np.array([point_pair[1] for point_pair in point_pairs])
    source_points = torch.tensor(source_points, dtype=torch.float32, requires_grad=False).cuda(0)
    target_points = torch.tensor(target_points, dtype=torch.float32, requires_grad=False).cuda(0)

    weight_tensor = torch.tensor(weight, dtype=torch.float32, requires_grad=False).unsqueeze(-1).cuda(0)

    matrix_00 = 1 + 1e-5
    matrix_01 = 1e-5
    matrix_02 = 1e-5
    matrix_10 = 1e-5
    matrix_11 = 1 + 1e-5
    matrix_12 = 1e-5
    matrix_20 = 1e-5
    matrix_21 = 1e-5
    matrix_22 = 1 + 1e-5

    transform_matrix = torch.tensor([[matrix_00, matrix_01, matrix_02],
                                     [matrix_10, matrix_11, matrix_12],
                                     [matrix_20, matrix_21, matrix_22]], dtype=torch.float32, requires_grad=True).cuda(0)
    transform_matrix = torch.nn.Parameter(transform_matrix)  # 将transform_matrix转换为一个可以优化的参数

    optimizer = torch.optim.Adam([transform_matrix], lr=learning_rate)
    last_error = 1000000
    grad_norm_list = []

    batch_groups = 4
    batch_size = int(np.ceil(len(source_points) / batch_groups))

    batch_source_points = []
    batch_target_points = []
    batch_weight_tensor = []
    for batch_index in range(batch_groups):
        start_index = batch_index * batch_size
        end_index = min((batch_index + 1) * batch_size, len(source_points))
        batch_source_points.append(source_points[start_index:end_index])
        batch_target_points.append(target_points[start_index:end_index])
        batch_weight_tensor.append(weight_tensor[start_index:end_index])

    square_scale_tensor = torch.tensor([1, 1, 1e4], dtype=torch.float32, requires_grad=False).cuda(0)
    minimal_error_transform_matrix = None
    minimal_error = 1000000
    minimal_error_iteration = 0
    for iteration_index in range(max_iterations):
        error_list = []
        for batch_index in range(batch_groups):
            start_index = batch_index * batch_size
            end_index = min((batch_index + 1) * batch_size, len(source_points))

            optimizer.zero_grad()
            batch_source_points_after_transform = torch.matmul(transform_matrix, batch_source_points[batch_index].transpose(0, 1)).transpose(0, 1)
            square = torch.square(batch_source_points_after_transform - batch_target_points[batch_index])
            square = square * square_scale_tensor
            square_with_weight = square * batch_weight_tensor[batch_index]
            error = torch.mean(square_with_weight)
            error_list.append(error.cpu().detach().numpy() * (end_index - start_index))
            error.backward()
            optimizer.step()

        if np.sum(error_list)/len(point_pairs) < minimal_error:
            minimal_error = np.sum(error_list)/len(point_pairs)
            minimal_error_transform_matrix = transform_matrix.cpu().detach().numpy()
            minimal_error_iteration = iteration_index

        print(f"iteration: {iteration_index}, error: {np.sum(error_list)/len(point_pairs)}, grad_norm: {torch.norm(transform_matrix.grad)}")
        # if iteration_index > 4:
        #     print(f"iteration: {iteration_index}, error: {np.sum(error_list)/len(point_pairs)}, transform_matrix: {transform_matrix}")
        #     if transform_matrix[2][0] > 1e-3:
        #         print()

        last_iteration_num = iteration_index
        grad_norm_list.append(torch.norm(transform_matrix.grad))

        # 如果500个iteration内都没有更新最小error，就停止。
        if minimal_error_iteration > iteration_index + 500:
            break

        # bool_stop = True
        # for grad_norm in grad_norm_list[-50:]:
        #     if grad_norm > stop_grad_norm:
        #         bool_stop = False
        #         break
        #
        # if last_iteration_num > 1000 and bool_stop:
        #     last_error = np.sum(error_list)/len(point_pairs)
        #     break
        # else:
        #     last_error = np.sum(error_list)/len(point_pairs)

    # transform_matrix = transform_matrix.cpu().detach().numpy()
    return minimal_error_transform_matrix, last_error, last_iteration_num


