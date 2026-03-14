import numpy as np
import heapq
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.backend_bases import MouseButton
import random
import astar
import time
def astar_search_3d_optimized_w(start_xyz, target_xyz, space_map,function_h_w:1.2):#function_h_w是启发函数中给h加上的权重的值
    # 转换为元组提高处理速度和作为字典键
    start = tuple(start_xyz)
    target = tuple(target_xyz)
    # 地图边界
    MAX_X, MAX_Y, MAX_Z = space_map.shape
    # 预计算26个方向的偏移量和距离，以存储空间来节省计算花费的时间
    neighbors_offsets = []
    for dx in [-1, 0, 1]:#遍历dx,dy,dz除了（0，0，0）外的其余26个组合并求解相应的距离
        for dy in [-1, 0, 1]:
            for dz in [-1, 0, 1]:
                if dx == 0 and dy == 0 and dz == 0:
                    continue
                # 计算到邻居的欧几里得距离作为移动代价
                dist = np.sqrt(dx**2 + dy**2 + dz**2)
                neighbors_offsets.append((dx, dy, dz, dist))
    # 优先队列 [f_score, g_score, (x, y, z)]
    #f_score表示的含义是从当前节点到目标点的距离，g_score表示的是走过的路程
    # heapq 会根据第一个元素 f_score 自动排序
    start_hn = np.linalg.norm(np.array(start) - np.array(target))#把tuple转化为np中的数组并求解三维距离
    #np.linalg.norm()函数用于计算向量的范数，如果是两个向量相减的话，求解的就是两点之间的欧几里得距离
    openset = [(start_hn, 0, start)]
    # 记录每个点是从哪个点来的 (用于回溯)
    came_from = {}
    # 记录从起点到当前点的实际代价 g
    g_score = {start: 0}
    # 记录是否已访问过 (Closed Set)
    closed_set = set()
    foundpath = 0
    final_node = None
    pop_print_flag=0
    pop_print_count=0
    while openset:
        # 1.弹出f最小的节点 (O(log N))
        t_pop_begin=time.time()
        current_f, current_g, current = heapq.heappop(openset)#由于heapq会按照第一个元素自动排序，所以每次弹出的都是f_score最小的节点
        t_pop_end=time.time()
        # if pop_print_flag==0 and pop_print_count>60:

        #     print(f"pop time cost of double optimized:{t_pop_end-t_pop_begin:.5f}")
        #     pop_print_flag=1
        #     pop_print_count=0
        pop_print_count+=1
        #print(f"pop time cost of double optimized:{t_pop_end-t_pop_begin:.5f}")
        #print(f"current pop count:{pop_print_count}")

        if current == target:#如果当前节点已经是目标节点，则说明找到了路径，设置foundpath为1并且记录当前节点为最终的节点
            foundpath = 1
            final_node = current
            break
        if current in closed_set:#如果当前节点已经访问过则跳过
            continue
        closed_set.add(current)#标记已经访问过该节点
        # 2. 遍历邻居
        cx, cy, cz = current#根据当前的current元组，获取cx,cy,cz的坐标值
        for dx, dy, dz, move_cost in neighbors_offsets:#遍历之前记录的neighbors_offsets中的26个偏移向量和距离
            nx, ny, nz = cx + dx, cy + dy, cz + dz
            neighbor = (nx, ny, nz)
            # 边界和障碍物检查
            if 0 <= nx < MAX_X and 0 <= ny < MAX_Y and 0 <= nz < MAX_Z:
                if space_map[nx, ny, nz] == 0: #这里设置space map中的0 是障碍物
                    continue                
                if neighbor in closed_set:#如果neighbor已经访问过则跳过
                    continue
                # 计算新的 g 值
                tentative_g = current_g + move_cost#计算新的g值，current_g值再加上移动的距离move_cost
                # dx1 = current[0] - target[0]
                # dy1 = current[1] - target[1]
                # dz1 = current[2] - target[2]
                # dx2 = start[0] - target[0]
                # dy2 = start[1] - target[1]
                # dz2 = start[2] - target[2]
                # # 叉乘的简化逻辑（在三维中即寻找偏离直线的程度），向量d1是当前点到目标点的向量，d2是起点到目标点的向量
                # cross_x = dy1 * dz2 - dy2 * dz1
                # cross_y = dx2 * dz1 - dx1 * dz2
                # cross_z = dx1 * dy2 - dx2 * dy1
                # dist_to_line = np.sqrt(cross_x**2 + cross_y**2 + cross_z**2)#求解点到直线距离的公式的分子，分母是起点到终点的欧几里得距离
                # 如果这个点没走过，或者找到了更短的路径
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    g_score[neighbor] = tentative_g
                    # 启发式函数 h (使用欧几里得距离)
                    h_score = np.sqrt((nx-target[0])**2 + (ny-target[1])**2 + (nz-target[2])**2)#q这里求解启发式函数，h_score表示的是邻近点到终点的距离
                    f_score = tentative_g + function_h_w*h_score#f_score表示的是从起点到当前点的距离再加上启发式函数h_score，这里给启发式函数的权重稍微大于1的目的是为了使得搜寻更加具有贪心性                
                    came_from[neighbor] = current#存储当前节点的上一个节点，父节点
                    heapq.heappush(openset, (f_score,tentative_g, neighbor))#用heapq把心增加的(f_score,g_score,neighbor)加入到openset中，heapq在推入后，会自动按照f_score进行排序
    # 3. 路径回溯
    path = []#存储路径
    if foundpath:#如果找到了路径
        curr = final_node#当前的节点则是最后的节点
        while curr in came_from:#如果curr在came_from,也即curr不是起点则继续回溯
            path.append(list(curr))#存储路径把Curr存储起来，作为回溯性的节点
            curr = came_from[curr]#获取当前一个节点的上一个节点，发挥类似链表的功能
        path.append(list(start))
        path.reverse() # A* 回溯路径从终点往回找，所以要反转
    return foundpath, path
