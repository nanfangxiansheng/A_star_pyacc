import numpy as np
import random

class RRTNode:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z
        self.parent = None

def rrt_search_3d(start_xyz, target_xyz, space_map, max_iter=2000, step_size=2.0, goal_sample_rate=0.1):
    """
    3D RRT 路径规划
    :param start_xyz: 起点 [x, y, z]
    :param target_xyz: 终点 [x, y, z]
    :param space_map: 3D numpy 数组
    :param max_iter: 最大迭代次数
    :param step_size: 每次扩展的步长
    :param goal_sample_rate: 采样目标点的概率（引导树向终点生长）
    """
    MAX_X, MAX_Y, MAX_Z = space_map.shape
    start_node = RRTNode(start_xyz[0], start_xyz[1], start_xyz[2])
    target_node = RRTNode(target_xyz[0], target_xyz[1], target_xyz[2])
    node_list = [start_node]

    def get_dist(n1, n2):
        return np.sqrt((n1.x - n2.x)**2 + (n1.y - n2.y)**2 + (n1.z - n2.z)**2)

    def is_collision_free(n1, n2):
        # 步长内插值检查碰撞
        dist = get_dist(n1, n2)
        steps = int(dist * 2) + 1 # 采样点密度
        for i in range(steps):
            u = i / steps
            curr_x = int(n1.x + u * (n2.x - n1.x))
            curr_y = int(n1.y + u * (n2.y - n1.y))
            curr_z = int(n1.z + u * (n2.z - n1.z))
            
            # 边界检查
            if not (0 <= curr_x < MAX_X and 0 <= curr_y < MAX_Y and 0 <= curr_z < MAX_Z):
                return False
            # 障碍物检查
            if space_map[curr_x, curr_y, curr_z] == 0:
                return False
        return True

    for i in range(max_iter):
        # 1. 采样随机点
        if random.random() < goal_sample_rate:
            rnd = [target_node.x, target_node.y, target_node.z]
        else:
            rnd = [random.uniform(0, MAX_X-1), random.uniform(0, MAX_Y-1), random.uniform(0, MAX_Z-1)]

        # 2. 找到树中最近的节点
        nearest_node = node_list[0]
        min_d = float('inf')
        for node in node_list:
            d = np.sqrt((node.x - rnd[0])**2 + (node.y - rnd[1])**2 + (node.z - rnd[2])**2)
            if d < min_d:
                min_d = d
                nearest_node = node

        # 3. 向随机点方向伸展 (Steer)
        theta = np.arctan2(rnd[1] - nearest_node.y, rnd[0] - nearest_node.x)
        dist_to_rnd = np.sqrt((rnd[0]-nearest_node.x)**2 + (rnd[1]-nearest_node.y)**2 + (rnd[2]-nearest_node.z)**2)
        phi = np.arccos((rnd[2] - nearest_node.z) / (dist_to_rnd if dist_to_rnd > 0 else 1))
        
        new_node = RRTNode(
            nearest_node.x + step_size * np.sin(phi) * np.cos(theta),
            nearest_node.y + step_size * np.sin(phi) * np.sin(theta),
            nearest_node.z + step_size * np.cos(phi)
        )

        # 4. 碰撞检查
        if is_collision_free(nearest_node, new_node):
            new_node.parent = nearest_node
            node_list.append(new_node)
            
            # 5. 检查是否接近目标
            if get_dist(new_node, target_node) <= step_size:
                if is_collision_free(new_node, target_node):
                    target_node.parent = new_node
                    
                    # 回溯生成路径
                    path = []
                    curr = target_node
                    while curr is not None:
                        path.append([curr.x, curr.y, curr.z])
                        curr = curr.parent
                    return True, path[::-1]

    return False, []