import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.backend_bases import MouseButton
import random

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
import time 
# 三维空间参数设置
MAX_X = 50
MAX_Y = 50
MAX_Z = 50

# 创建三维空间障碍物
def create_3d_obstacles():
    # 初始化空间，1表示可通行，0表示障碍物
    space = np.ones((MAX_X, MAX_Y, MAX_Z))
    
    # 创建一些随机障碍物
    num_obstacles = 15
    for _ in range(num_obstacles):
        # 随机生成障碍物中心点
        center_x = random.randint(5, MAX_X-5)
        center_y = random.randint(5, MAX_Y-5)
        center_z = random.randint(5, MAX_Z-5)
        
        # 随机生成障碍物大小
        size_x = random.randint(3, 10)
        size_y = random.randint(3, 10)
        size_z = random.randint(3, 10)
        
        # 创建立方体障碍物
        for x in range(max(0, center_x-size_x//2), min(MAX_X, center_x+size_x//2)):
            for y in range(max(0, center_y-size_y//2), min(MAX_Y, center_y+size_y//2)):
                for z in range(max(0, center_z-size_z//2), min(MAX_Z, center_z+size_z//2)):
                    space[x, y, z] = 0
    
    return space

# 距离函数
def distanceFcn(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

# A*算法函数 - 三维版本
def astar_search_3d(start_xyz, target_xyz, space_map):
    # AStar算法初始化
    GlbTab = np.zeros((MAX_X, MAX_Y, MAX_Z))  # 0|new 1|open 2|close
    PathTab = np.zeros((MAX_X, MAX_Y, MAX_Z, 3))
    nodeStartXYZ = start_xyz
    nodeTargetXYZ = target_xyz
    startGn = 0
    startHn = distanceFcn(nodeTargetXYZ, nodeStartXYZ)
    startFn = startGn + startHn
    # [fn | gn | hn | x | y | z]
    nodeStart = [startFn, startGn, startHn, nodeStartXYZ[0], nodeStartXYZ[1], nodeStartXYZ[2]]

    # 主循环
    openset = [nodeStart]
    foundpath = 0

    while openset:
        # 找到f值最小的节点
        minIdx = np.argmin([node[0] for node in openset])
        node = openset[minIdx]
        openset.pop(minIdx)
        
        if [node[3], node[4], node[5]] == nodeTargetXYZ:
            foundpath = 1
            break
        
        node_x = int(node[3])
        node_y = int(node[4])
        node_z = int(node[5])
        node_gn = node[1]
        GlbTab[node_x, node_y, node_z] = 2
        
        # 检查26个相邻节点（三维空间）
        for k in range(-1, 2):
            for j in range(-1, 2):
                for i in range(-1, 2):
                    # 跳过当前节点
                    if k == 0 and j == 0 and i == 0:
                        continue
                    
                    s_x = node_x + k
                    s_y = node_y + j
                    s_z = node_z + i
                    
                    # 检查边界
                    if (0 <= s_x < MAX_X) and (0 <= s_y < MAX_Y) and (0 <= s_z < MAX_Z):
                        # 检查是否是障碍物或已关闭节点
                        if GlbTab[s_x, s_y, s_z] == 2 or space_map[s_x, s_y, s_z] == 0:
                            continue
                        
                        # 计算新节点的代价
                        s_gn = node_gn + distanceFcn([node_x, node_y, node_z], [s_x, s_y, s_z])
                        s_hn = distanceFcn(nodeTargetXYZ, [s_x, s_y, s_z])
                        s_fn = s_gn + s_hn
                        
                        if GlbTab[s_x, s_y, s_z] == 0:
                            # 新节点
                            GlbTab[s_x, s_y, s_z] = 1
                            openset.append([s_fn, s_gn, s_hn, s_x, s_y, s_z])
                            PathTab[s_x, s_y, s_z] = [node_x, node_y, node_z]
                        else:
                            # 存在开放节点
                            existIdx = None
                            for idx, open_node in enumerate(openset):
                                if open_node[3] == s_x and open_node[4] == s_y and open_node[5] == s_z:
                                    existIdx = idx
                                    break
                            
                            if existIdx is not None:
                                exist_gn = openset[existIdx][1]
                                if exist_gn > s_gn:
                                    openset[existIdx] = [s_fn, s_gn, s_hn, s_x, s_y, s_z]
                                    PathTab[s_x, s_y, s_z] = [node_x, node_y, node_z]

    # 获取路径
    path = []
    if foundpath == 1:
        node_xyz = nodeTargetXYZ.copy()
        path.append(node_xyz.copy())
        
        while node_xyz != nodeStartXYZ:
            node_xyz = PathTab[int(node_xyz[0]), int(node_xyz[1]), int(node_xyz[2])].astype(int)
            node_xyz = node_xyz.tolist()
            path.append(node_xyz.copy())
    
    return foundpath, path

# 可视化三维空间和路径
def visualize_3d_path(start_xyz, target_xyz, path, space_map):
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # 绘制障碍物
    obstacle_positions = np.where(space_map == 0)
    ax.scatter(obstacle_positions[0], obstacle_positions[1], obstacle_positions[2], 
               color='gray', alpha=0.3, marker='s', label='障碍物')
    
    # 绘制起点和终点
    ax.scatter(start_xyz[0], start_xyz[1], start_xyz[2], color='blue', s=100, label='起点')
    ax.scatter(target_xyz[0], target_xyz[1], target_xyz[2], color='red', s=100, label='终点')
    
    # 绘制路径
    if path:
        path_array = np.array(path)
        ax.plot(path_array[:, 0], path_array[:, 1], path_array[:, 2], 'g-', linewidth=2, label='路径')
        ax.scatter(path_array[:, 0], path_array[:, 1], path_array[:, 2], color='yellow', s=20)
    
    # 设置坐标轴标签和标题
    ax.set_xlabel('X轴')
    ax.set_ylabel('Y轴')
    ax.set_zlabel('Z轴')
    ax.set_title('三维A*路径规划结果')
    
    # 设置坐标轴范围
    ax.set_xlim(0, MAX_X)
    ax.set_ylim(0, MAX_Y)
    ax.set_zlim(0, MAX_Z)
    
    # 添加图例
    ax.legend()
    
    plt.tight_layout()
    plt.show()

# 交互式选择起点和终点
def select_points_3d(space_map):
    fig = plt.figure(figsize=(15, 5))
    
    # 创建三个子图：XY平面、XZ平面和YZ平面
    ax1 = fig.add_subplot(131)
    ax2 = fig.add_subplot(132)
    ax3 = fig.add_subplot(133)
    
    # 显示三个平面的障碍物投影
    xy_projection = np.min(space_map, axis=2)  # XY平面投影
    xz_projection = np.min(space_map, axis=1)  # XZ平面投影
    yz_projection = np.min(space_map, axis=0)  # YZ平面投影
    
    ax1.imshow(xy_projection.T, origin='lower', cmap='gray')
    ax1.set_title('XY平面 (选择X和Y)')
    ax1.set_xlabel('X轴')
    ax1.set_ylabel('Y轴')
    
    ax2.imshow(xz_projection.T, origin='lower', cmap='gray')
    ax2.set_title('XZ平面 (选择Z)')
    ax2.set_xlabel('X轴')
    ax2.set_ylabel('Z轴')
    
    ax3.imshow(yz_projection.T, origin='lower', cmap='gray')
    ax3.set_title('YZ平面')
    ax3.set_xlabel('Y轴')
    ax3.set_ylabel('Z轴')
    
    points = {'start': None, 'target': None}
    current_point = 'start'
    
    def onclick(event):
        nonlocal current_point
        
        if event.inaxes == ax1 and event.button == MouseButton.LEFT:
            x, y = int(event.xdata), int(event.ydata)
            
            if current_point == 'start':
                points['start'] = [x, y, None]
                ax1.plot(x, y, 'bo', markersize=10)
                ax1.set_title('已选择起点XY坐标，请在XZ平面选择起点Z坐标')
            else:
                points['target'] = [x, y, None]
                ax1.plot(x, y, 'ro', markersize=10)
                ax1.set_title('已选择终点XY坐标，请在XZ平面选择终点Z坐标')
            
            fig.canvas.draw()
            
        elif event.inaxes == ax2 and event.button == MouseButton.LEFT:
            x, z = int(event.xdata), int(event.ydata)
            
            if current_point == 'start' and points['start'] is not None:
                if x == points['start'][0]:  # 确保X坐标一致
                    points['start'][2] = z
                    ax2.plot(x, z, 'bo', markersize=10)
                    ax2.set_title('已选择起点Z坐标')
                    current_point = 'target'
                    ax1.set_title('请在XY平面选择终点XY坐标')
                else:
                    ax2.set_title('X坐标不匹配，请重新选择')
            
            elif current_point == 'target' and points['target'] is not None:
                if x == points['target'][0]:  # 确保X坐标一致
                    points['target'][2] = z
                    ax2.plot(x, z, 'ro', markersize=10)
                    ax2.set_title('已选择终点Z坐标')
                    
                    # 检查是否所有点都已选择
                    if None not in points['start'] and None not in points['target']:
                        plt.close(fig)
                else:
                    ax2.set_title('X坐标不匹配，请重新选择')
            
            fig.canvas.draw()
    
    # 连接鼠标点击事件
    cid = fig.canvas.mpl_connect('button_press_event', onclick)
    plt.tight_layout()
    plt.show()
    
    # 确保所有坐标都已选择
    if None in points['start'] or None in points['target']:
        return None, None
    
    return points['start'], points['target']

# 主函数
def main():
    print("正在创建三维空间环境...")
    space_map = create_3d_obstacles()
    
    print("请在图形界面中选择起点和终点：")
    start_point, target_point = select_points_3d(space_map)
    
    if start_point and target_point:
        print(f"起点坐标: {start_point}")
        print(f"终点坐标: {target_point}")
        
        # 检查起点和终点是否在障碍物上
        if space_map[start_point[0], start_point[1], start_point[2]] == 0:
            print("错误：起点位于障碍物上！")
            return
        
        if space_map[target_point[0], target_point[1], target_point[2]] == 0:
            print("错误：终点位于障碍物上！")
            return
        
        print("正在计算路径...")
        start_time=time.time()
        found, path = astar_search_3d(start_point, target_point, space_map)
        end_time=time.time()
        print("It took %.2f seconds to find the path." % (end_time - start_time))
        
        if found:
            print("路径已找到！正在可视化...")
            visualize_3d_path(start_point, target_point, path, space_map)
        else:
            print("未找到路径！")
    else:
        print("未完成点选择，程序退出。")

# 运行主函数
if __name__ == "__main__":
    main()