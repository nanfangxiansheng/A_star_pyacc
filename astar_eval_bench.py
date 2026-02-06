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
from utils import *
import scipy
from rrt_original import rrt_search_3d
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

# --- 4. 自动化基准测试函数 ---
def run_benchmark(num_tests=20,save_root='.\\results\\'):
    print(f"开始性能测试，总次数: {num_tests}...")
    original_times = []
    optimized_times = []
    single_optimized_times = []
    rrt_times=[]
    rrt_origin_times=[]
    len_optimized_paths=[]
    len_rrt_star=[]
    speedups=[]
    len_rrt_origin_paths=[]

    single_speedups=[]

    rrt_related_speedups=[]

    rrt_origin_related_speedups=[]
    success_count = 0

    for i in range(num_tests):
        os.makedirs(save_root+f"{i+1}_out",exist_ok=True)
        space_map = create_3d_obstacles()
        start_pt, target_pt = get_random_points(space_map, min_dist=25)
        print(f"found_pt: {start_pt}, target_pt: {target_pt}")
        
        # 测试两次加速版本
        t0 = time.time()
        found_opt,path1 = astar_search_3d_optimized_w(start_pt, target_pt, space_map)
        path_0=path1
        t_opt = time.time() - t0

        trajectory_1=time.time()
        path1=b_spline_optimization(path1)#轨迹优化
        path1=list(path1)
        trajectory_cost=time.time()-trajectory_1
        print(f"轨迹优化时间: {trajectory_cost}")
        visualize_3d_path_save(start_pt,target_pt,path1,space_map,save_path=save_root+f"{i+1}_out\\{i+1}_acc_b.png")
        
        visualize_3d_path_save(start_pt,target_pt,path_0,space_map,save_path=save_root+f"{i+1}_out\\{i+1}_acc.png")
        # 测试单次加速版本
        t2=time.time()
        found_orig,path2 = astar_search_3d_optimized(start_pt, target_pt, space_map)
        t_single = time.time() - t2
        visualize_3d_path_save(start_pt,target_pt,path2,space_map,save_path=save_root+f'{i+1}_out\\{i+1}_easy_acc.png')
        # 测试原始版
        t1 = time.time()
        found_non,path3=astar_search_3d(start_pt, target_pt, space_map)
        t_orig = time.time() - t1

        visualize_3d_path_save(start_pt,target_pt,path3,space_map,save_path=save_root+f'{i+1}_out\\{i+1}_origin.png')

        #测试RRT star方案
        t3=time.time()
        found_RRTstar,path4=rrt_star_3d(start_pt, target_pt, space_map)#调用RRT*
        cost_rrt=time.time()-t3

        #测试RRT原始的方案
        t4=time.time()
        found_rrt,path5=rrt_search_3d(start_pt, target_pt, space_map)#调用RRT算法
        cost_rrt_origin=time.time()-t4
        if found_RRTstar:
            print("RRT*成功")
            print(f"RRT*耗时:{cost_rrt}")
            visualize_3d_path_save(start_pt,target_pt,path4,space_map,save_path=save_root+f'{i+1}_out\\{i+1}_rrt.png')
            rrt_times.append(cost_rrt)

        if found_rrt:
            print("RRT成功")
            print(f"RRT耗时:{cost_rrt_origin}")
            visualize_3d_path_save(start_pt,target_pt,path5,space_map,save_path=save_root+f'{i+1}_out\\{i+1}_rrt_origin.png')
            rrt_origin_times.append(cost_rrt_origin)


        len_acc_b=get_paths_total_length(path1)#b优化后的轨迹
        len_acc=get_paths_total_length(path_0)
        len_easy_acc=get_paths_total_length(path2)
        len_origin=get_paths_total_length(path3)
        len_rrt=get_paths_total_length(path4)
        len_rrt_origin=get_paths_total_length(path5)#获得RRT origin的路径长度

        len_optimized_paths.append(len_acc_b)
        len_rrt_origin_paths.append(len_rrt_origin)
        len_rrt_star.append(len_rrt)
        print(f"len_acc_b:{len_acc_b},len_acc:{len_acc},len_easy_acc:{len_easy_acc},len_origin:{len_origin},len_rrt_star:{len_rrt},len_rrt_origin:{len_rrt_origin}")
        if t_opt<=0.001:
            t_opt=0.001
        # 只有当路径确实存在时，对比才有意义（虽然算法失败耗时也值得参考）
        if found_opt and found_orig:
            success_count += 1
            original_times.append(t_orig)
            optimized_times.append(t_opt)
            single_optimized_times.append(t_single)
            speedups.append(t_orig/t_opt)
            single_speedups.append(t_orig/t_single)
            rrt_related_speedups.append(cost_rrt/t_opt)
            rrt_origin_related_speedups.append(cost_rrt_origin/t_opt)
            print(f"测试 {i+1:02d}: 原始 {t_orig:.4f}s |单次优化{t_single:.4f}s| 二次优化 {t_opt:.4f}s | 二次加速 {t_orig/t_opt:.2f}x|单次加速{t_orig/t_single:.2f}x")
        else:
            print(f"测试 {i+1:02d}: 路径不可达，跳过数据统计")

    if success_count > 0:
        avg_orig = np.mean(original_times)
        avg_opt = np.mean(optimized_times)
        avg_single_opt = np.mean(single_optimized_times)
        avg_rrt=np.mean(rrt_times)
        avg_rrt_origin=np.mean(rrt_origin_times)
        avg_len_optimized_paths=np.mean(len_optimized_paths)
        avg_len_rrt_star=np.mean(len_rrt_star)
        avg_len_rrt_origin=np.mean(len_rrt_origin_paths)

        print("\n" + "="*30)
        print(f"测试完成！成功找到路径次数: {success_count}")
        print(f"原始算法平均耗时: {avg_orig:.4f} s")
        print(f"优化算法平均耗时: {avg_opt:.4f} s")
        print(f"平均二次加速比: {avg_orig/avg_opt:.2f} 倍")
        print(f"平均一次加速比:{avg_orig/avg_single_opt:.2f}倍")
        print(f"平均RRT*耗时:{avg_rrt:.4f}")
        print(f"平均RRT origin耗时:{avg_rrt_origin:.4f}")
        print(f"对比于RRT*的加速比:{avg_rrt/avg_opt:.2f}")
        print(f"对比于RRT的加速比:{avg_rrt_origin/avg_opt:.2f}")
        print(f"average_len_optimized_paths:{avg_len_optimized_paths:.4f}")
        print(f"average_len_rrt_star:{avg_len_rrt_star:.4f}")
        print(f"average_len_rrt_origin:{avg_len_rrt_origin:.4f}")
        print(f"路径缩减比例(相对于RRT*)：{avg_len_rrt_star/avg_len_optimized_paths:.4f}")
        print(f"路径缩减比例(相对于RRT origin)：{avg_len_rrt_origin/avg_len_optimized_paths:.4f}")

        
        print("="*30)
        plot_benchmark_results(rrt_origin_times, optimized_times, rrt_origin_related_speedups)

    else:
        print("所有测试均未找到有效路径，请调整障碍物密度或地图大小。")

# 运行主函数
if __name__ == "__main__":
    #main()
    save_root='.\\results\\'
    os.makedirs(save_root, exist_ok=True)
    run_benchmark(save_root=save_root)