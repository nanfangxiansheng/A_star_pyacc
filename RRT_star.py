import numpy as np
import random

class RRTNode:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z
        self.parent = None
        self.cost = 0.0

def rrt_star_3d(start_xyz, target_xyz, space_map, max_iter=1500, step_size=2.0, search_radius=5.0, goal_sample_rate=0.1):
    """
    3D RRT* 路径规划实现
    :param start_xyz: 起点 [x, y, z]
    :param target_xyz: 终点 [x, y, z]
    :param space_map: 3D numpy 数组 (1为可通行, 0为障碍物)
    :param max_iter: 最大迭代次数
    :param step_size: 步长
    :param search_radius: 邻域搜索半径 (Rewire使用)
    :param goal_sample_rate: 采样目标点的概率 (Goal Bias)
    """
    MAX_X, MAX_Y, MAX_Z = space_map.shape
    start_node = RRTNode(start_xyz[0], start_xyz[1], start_xyz[2])
    target_node = RRTNode(target_xyz[0], target_xyz[1], target_xyz[2])
    nodes = [start_node]

    def get_dist(n1, n2):
        return np.sqrt((n1.x - n2.x)**2 + (n1.y - n2.y)**2 + (n1.z - n2.z)**2)

    def is_collision_free(n1, n2):
        # 简单的线性插值碰撞检测
        points_count = int(get_dist(n1, n2) * 2) + 2
        for i in range(points_count):
            u = i / (points_count - 1)
            curr_x = int(n1.x + u * (n2.x - n1.x))
            curr_y = int(n1.y + u * (n2.y - n1.y))
            curr_z = int(n1.z + u * (n2.z - n1.z))
            
            if not (0 <= curr_x < MAX_X and 0 <= curr_y < MAX_Y and 0 <= curr_z < MAX_Z):
                return False
            if space_map[curr_x, curr_y, curr_z] == 0:
                return False
        return True

    for i in range(max_iter):
        # 1. 采样
        if random.random() < goal_sample_rate:
            rnd_point = [target_node.x, target_node.y, target_node.z]
        else:
            rnd_point = [random.uniform(0, MAX_X-1), random.uniform(0, MAX_Y-1), random.uniform(0, MAX_Z-1)]

        # 2. 找到最近节点
        nearest_node = nodes[0]
        min_dist = float('inf')
        for node in nodes:
            dist = np.sqrt((node.x - rnd_point[0])**2 + (node.y - rnd_point[1])**2 + (node.z - rnd_point[2])**2)
            if dist < min_dist:
                min_dist = dist
                nearest_node = node

        # 3. 伸展 (Steer)
        theta = np.arctan2(rnd_point[1] - nearest_node.y, rnd_point[0] - nearest_node.x)
        phi = np.arccos((rnd_point[2] - nearest_node.z) / (min_dist if min_dist > 0 else 1))
        
        # 限制步长
        actual_step = min(step_size, min_dist)
        new_node = RRTNode(
            nearest_node.x + actual_step * np.sin(phi) * np.cos(theta),
            nearest_node.y + actual_step * np.sin(phi) * np.sin(theta),
            nearest_node.z + actual_step * np.cos(phi)
        )

        if not is_collision_free(nearest_node, new_node):
            continue

        # 4. 找到搜索半径内的邻居节点 (RRT* 核心：Choose Parent)
        neighbors = []
        for node in nodes:
            if get_dist(node, new_node) < search_radius:
                neighbors.append(node)

        # 选择代价最小的父节点
        best_parent = nearest_node
        min_cost = nearest_node.cost + get_dist(nearest_node, new_node)
        
        for neighbor in neighbors:
            if is_collision_free(neighbor, new_node):
                new_cost = neighbor.cost + get_dist(neighbor, new_node)
                if new_cost < min_cost:
                    min_cost = new_cost
                    best_parent = neighbor
        
        new_node.parent = best_parent
        new_node.cost = min_cost
        nodes.append(new_node)

        # 5. 重布线 (Rewire)
        for neighbor in neighbors:
            if is_collision_free(new_node, neighbor):
                new_cost = new_node.cost + get_dist(new_node, neighbor)
                if new_cost < neighbor.cost:
                    neighbor.parent = new_node
                    neighbor.cost = new_cost

        # 检查是否接近目标
        if get_dist(new_node, target_node) < step_size:
            if is_collision_free(new_node, target_node):
                target_node.parent = new_node
                target_node.cost = new_node.cost + get_dist(new_node, target_node)
                
                # 回溯路径
                path = []
                curr = target_node
                while curr is not None:
                    path.append([curr.x, curr.y, curr.z])
                    curr = curr.parent
                return True, path[::-1]

    return False, []