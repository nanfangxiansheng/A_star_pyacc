import numpy as np
import heapq
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.backend_bases import MouseButton
import random
import astar
from astar import astar_search_3d
from RRT_star import rrt_star_3d
from astar_optimed_double import astar_search_3d_optimized_w
from astar_optimized_single import astar_search_3d_optimized
import scipy
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

import sys
from scipy import interpolate

def b_spline_optimization(path, n_points=200, degree=3, smoothing=2.0):
    """
    使用B样条对路径进行平滑优化
    :param path: A*算法生成的原始路径点列表 [(x1,y1,z1), (x2,y2,z2), ...]
    :param n_points: 插值生成的点数量，越多越平滑
    :param degree: 样条阶数，k=3表示三次样条（常用）
    :param smoothing: 平滑因子（s）。s=0表示必须经过所有原始点；s越大曲线越平滑但偏离原始点越远
    :return: 优化后的平滑路径数组
    ```python
    """
    if len(path) <= degree:
        return np.array(path)

    # 1. 数据预处理
    path_arr = np.array(path)
    
    # B样条要求输入点不能有重复点（A*路径有时会出现相邻重复点）
    # 过滤掉连续重复的点
    filtered_path = [path_arr[0]]
    for i in range(1, len(path_arr)):
        if not np.allclose(path_arr[i], path_arr[i-1]):
            filtered_path.append(path_arr[i])
    path_arr = np.array(filtered_path)

    if len(path_arr) <= degree:
        return path_arr

    # 2. 拟合 B-Spline
    # x, y, z 分别为各轴坐标
    x, y, z = path_arr[:, 0], path_arr[:, 1], path_arr[:, 2]
    
    # splprep 用于寻找多维空间的参数化曲线
    # tck: 包含节点向量、系数、阶数的元组; u: 参数化后的值
    try:
        tck, u = interpolate.splprep([x, y, z], s=smoothing, k=degree)
        
        # 3. 在 0 到 1 之间生成更密集的参数点来评估曲线
        u_fine = np.linspace(0, 1, n_points)
        new_points = interpolate.splev(u_fine, tck)
        
        return np.array(new_points).T
    except Exception as e:
        print(f"样条拟合失败: {e}")
        return path_arr
# 增加递归深度，防止3D长距离跳跃时溢出


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
def get_paths_total_length(path):#获得一段path的总长度

    return sum([distanceFcn(path[i], path[i+1]) for i in range(len(path)-1)])


def plot_benchmark_results(original_times, optimized_times, speedups):
    fig = plt.figure(figsize=(15, 5))
    
    # 1. 耗时对比柱状图
    ax1 = fig.add_subplot(131)
    indices = np.arange(len(original_times))
    width = 0.35
    ax1.bar(indices - width/2, original_times, width, label='RRT算法', color='salmon')
    ax1.bar(indices + width/2, optimized_times, width, label='优化A*算法', color='skyblue')
    ax1.set_xlabel('测试序号')
    ax1.set_ylabel('耗时 (秒)')
    ax1.set_title('各次测试耗时对比')
    ax1.legend()
    ax1.grid(axis='y', linestyle='--', alpha=0.7)

    # 2. 加速比折线图
    ax2 = fig.add_subplot(132)
    ax2.plot(indices, speedups, marker='o', linestyle='-', color='green')
    #ax2.axhline(y=np.mean(speedups), color='red', linestyle='--', label=f'平均: {np.mean(speedups):.1f}x')
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
        #path=b_spline_optimization(path)#贪心路径修剪

        end_time1=time.time()
        print("It took %.2f seconds to find the path.(acc version)" % (end_time1 - start_time1))

        start_time2=time.time()
        found1,path1=astar_search_3d_optimized_w(start_point, target_point, space_map)
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
