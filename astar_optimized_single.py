import numpy as np
import numpy as np
import heapq
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.backend_bases import MouseButton
import random
import astar
def astar_search_3d_optimized(start_xyz, target_xyz, space_map):
    # 转换为元组提高处理速度和作为字典键
    start = tuple(start_xyz)
    target = tuple(target_xyz)
    
    # 地图边界
    MAX_X, MAX_Y, MAX_Z = space_map.shape

    # 预计算26个方向的偏移量和距离，以存储空间来节省计算花费的时间
    neighbors_offsets = []
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            for dz in [-1, 0, 1]:
                if dx == 0 and dy == 0 and dz == 0:
                    continue
                # 计算到邻居的欧几里得距离作为移动代价
                dist = np.sqrt(dx**2 + dy**2 + dz**2)
                neighbors_offsets.append((dx, dy, dz, dist))

    # 优先队列 [f_score, g_score, (x, y, z)]
    # heapq 会根据第一个元素 f_score 自动排序
    start_hn = np.linalg.norm(np.array(start) - np.array(target))
    openset = [(start_hn, 0, start)]
    
    # 记录每个点是从哪个点来的 (用于回溯)
    came_from = {}
    
    # 记录从起点到当前点的实际代价 g
    g_score = {start: 0}
    
    # 记录是否已访问过 (Closed Set)
    closed_set = set()
    foundpath = 0
    final_node = None
    while openset:
        # 1. 弹出 f 最小的节点 (O(log N))
        current_f, current_g, current = heapq.heappop(openset)
        if current == target:
            foundpath = 1
            final_node = current
            break
        if current in closed_set:
            continue
        closed_set.add(current)
        # 2. 遍历邻居
        cx, cy, cz = current
        for dx, dy, dz, move_cost in neighbors_offsets:
            nx, ny, nz = cx + dx, cy + dy, cz + dz
            neighbor = (nx, ny, nz)
            # 边界和障碍物检查
            if 0 <= nx < MAX_X and 0 <= ny < MAX_Y and 0 <= nz < MAX_Z:
                if space_map[nx, ny, nz] == 0: # 假设 0 是障碍物
                    continue                
                if neighbor in closed_set:
                    continue
                # 计算新的 g 值
                tentative_g = current_g + move_cost
                # 如果这个点没走过，或者找到了更短的路径
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    g_score[neighbor] = tentative_g
                    # 启发式函数 h (使用欧几里得距离)
                    h_score = np.sqrt((nx-target[0])**2 + (ny-target[1])**2 + (nz-target[2])**2)
                    f_score = tentative_g +h_score                 
                    came_from[neighbor] = current
                    heapq.heappush(openset, (f_score, tentative_g, neighbor))
    # 3. 路径回溯
    path = []
    if foundpath:
        curr = final_node
        while curr in came_from:
            path.append(list(curr))
            curr = came_from[curr]
        path.append(list(start))
        path.reverse() # A* 通常从终点往回找，所以要反转
    return foundpath, path