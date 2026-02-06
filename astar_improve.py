import numpy as np
import heapq
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.backend_bases import MouseButton
import random
import astar
from astar import astar_search_3d
# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
import time 
# 三维空间参数设置
MAX_X = 50
MAX_Y = 50
MAX_Z = 50
import os
random.seed(42)#设置初始随机种子
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
                    f_score = tentative_g + h_score
                    
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

def visualize_3d_path_save(start_xyz, target_xyz, path, space_map,save_path):
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
    plt.savefig(save_path)
    plt.close()
    #plt.show()

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
def get_random_points(space_map, min_dist=40):
    """随机选择不在障碍物上且距离大于阈值的两点"""
    while True:
        s = [random.randint(0, MAX_X-1) for _ in range(3)]
        t = [random.randint(0, MAX_X-1) for _ in range(3)]
        
        if space_map[s[0], s[1], s[2]] == 1 and space_map[t[0], t[1], t[2]] == 1:
            dist = np.linalg.norm(np.array(s) - np.array(t))
            if dist >= min_dist and abs(s[2]-t[2])>=4:
                return s, t

# --- 4. 自动化基准测试函数 ---
def run_benchmark(num_tests=30,save_root='.\\results\\'):
    print(f"开始性能测试，总次数: {num_tests}...")
    original_times = []
    optimized_times = []
    speedups=[]
    success_count = 0

    for i in range(num_tests):
        os.makedirs(save_root+f"{i+1}_out",exist_ok=True)
        space_map = create_3d_obstacles()
        start_pt, target_pt = get_random_points(space_map, min_dist=25)
        print(f"found_pt: {start_pt}, target_pt: {target_pt}")
        
        # 测试优化版
        t0 = time.time()
        found_opt,path1 = astar_search_3d_optimized(start_pt, target_pt, space_map)
        visualize_3d_path_save(start_pt,target_pt,path1,space_map,save_path=save_root+f"{i+1}_out\\{i+1}_acc.png")
        t_opt = time.time() - t0
        
        # 测试原始版
        t1 = time.time()
        found_orig,path2 = astar_search_3d(start_pt, target_pt, space_map)
        visualize_3d_path_save(start_pt,target_pt,path1,space_map,save_path=save_root+f'{i+1}_out\\{i+1}_origin.png')

        t_orig = time.time() - t1
        
        # 只有当路径确实存在时，对比才有意义（虽然算法失败耗时也值得参考）
        if found_opt and found_orig:
            success_count += 1
            original_times.append(t_orig)
            optimized_times.append(t_opt)
            speedups.append(t_orig/t_opt)
            print(f"测试 {i+1:02d}: 原始 {t_orig:.4f}s | 优化 {t_opt:.4f}s | 加速 {t_orig/t_opt:.2f}x")
        else:
            print(f"测试 {i+1:02d}: 路径不可达，跳过数据统计")

    if success_count > 0:
        avg_orig = np.mean(original_times)
        avg_opt = np.mean(optimized_times)

        print("\n" + "="*30)
        print(f"测试完成！成功找到路径次数: {success_count}")
        print(f"原始算法平均耗时: {avg_orig:.4f} s")
        print(f"优化算法平均耗时: {avg_opt:.4f} s")
        print(f"平均加速比: {avg_orig/avg_opt:.2f} 倍")
        print("="*30)
        plot_benchmark_results(original_times, optimized_times, speedups)

    else:
        print("所有测试均未找到有效路径，请调整障碍物密度或地图大小。")
def plot_benchmark_results(original_times, optimized_times, speedups):
    fig = plt.figure(figsize=(15, 5))
    
    # 1. 耗时对比柱状图
    ax1 = fig.add_subplot(131)
    indices = np.arange(len(original_times))
    width = 0.35
    ax1.bar(indices - width/2, original_times, width, label='原始算法', color='salmon')
    ax1.bar(indices + width/2, optimized_times, width, label='优化算法', color='skyblue')
    ax1.set_xlabel('测试序号')
    ax1.set_ylabel('耗时 (秒)')
    ax1.set_title('各次测试耗时对比')
    ax1.legend()
    ax1.grid(axis='y', linestyle='--', alpha=0.7)

    # 2. 加速比折线图
    ax2 = fig.add_subplot(132)
    ax2.plot(indices, speedups, marker='o', linestyle='-', color='green')
    ax2.axhline(y=np.mean(speedups), color='red', linestyle='--', label=f'平均: {np.mean(speedups):.1f}x')
    ax2.set_xlabel('测试序号')
    ax2.set_ylabel('加速倍数')
    ax2.set_title('加速效率分布')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. 加速比箱线图 (展示稳定性)
    ax3 = fig.add_subplot(133)
    ax3.boxplot(speedups, patch_artist=True, boxprops=dict(facecolor='lightgreen'))
    ax3.set_title('加速比统计分布')
    ax3.set_ylabel('加速倍数')
    ax3.set_xticklabels(['A* 优化效果'])

    plt.tight_layout()
    plt.show()
    plt.savefig('benchmark_results.png')
    plt.close()
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
        start_time1=time.time()
        found, path = astar_search_3d_optimized(start_point, target_point, space_map)
        end_time1=time.time()
        print("It took %.2f seconds to find the path.(acc version)" % (end_time1 - start_time1))

        start_time2=time.time()
        found1,path1=astar_search_3d(start_point, target_point, space_map)
        end_time2=time.time()
        print("It took %.2f seconds to find the path.(original version)" % (end_time2 - start_time2))
        
        if found:
            print("路径已找到！正在可视化...")
            visualize_3d_path(start_point, target_point, path, space_map)
            print("---------")
            print("现在开始可视化路径二")
            visualize_3d_path(start_point, target_point, path1, space_map)

        else:
            print("未找到路径！")
    else:
        print("未完成点选择，程序退出。")

# 运行主函数
if __name__ == "__main__":
    #main()
    save_root='.\\results\\'
    os.makedirs(save_root, exist_ok=True)
    run_benchmark(save_root=save_root)