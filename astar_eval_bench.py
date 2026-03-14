import numpy as np
import heapq
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D#mplot3d是matplotlib的一个工具包，提供了3D绘图功能
from matplotlib.backend_bases import MouseButton#鼠标点击事件
import random
import astar#导入astar模块
from astar import astar_search_3d#导入astar模块中的astar_search_3d函数
from astar import astar_search_3d_with_data_structure_optimization,astar_search_3d_with_precomputed_neighbors
from RRT_star import rrt_star_3d#导入了rrt_star模块
from astar_optimed_double import astar_search_3d_optimized_w#导入了astar_search_optimized_double模块，加上w表示其给启发函数加上了w权重
from astar_optimized_single import astar_search_3d_optimized#导入了astar_search_optimized_single模块，数据结构优化模块
from utils import *
import scipy
from rrt_original import rrt_search_3d#导入了rrt original模块
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
def run_benchmark(num_tests=10,save_root='.\\results\\'):
    print(f"开始性能测试，总次数: {num_tests}...")
    original_times = []#原始的astar的算法时间
    optimized_times = []#astar数据结构加上启发函数权重优化的时间(默认w的值是1.2)
    single_optimized_times = []#astar数据结构优化的时间
    rrt_times=[]#rrt*算法的时间
    rrt_origin_times=[]#RRT算法的花费时间
    len_optimized_paths=[]#astar数据结构优化后的路径长度
    len_rrt_star=[]#rrt*算法路径长度
    speedups=[]#astar数据结构优化加上启发函数权重优化后的加速倍数
    len_rrt_origin_paths=[]#rrt原始算法的路径长度
    precompue_neighbor_times=[]#预先计算邻近值的时间
    data_structure_optimization_times=[]#数据结构优化的时间
    single_speedups=[]#astar数据结构优化后的加速倍数
    precompute_neighbors_speedups=[]#预先计算邻近值导致的加速倍数
    data_structure_optimization_speedups=[]#数据结构优化导致的加速倍数
    rrt_related_speedups=[]#相比较于rrt*的加速倍数
    rrt_origin_related_speedups=[]#相比较于rrt原始算法的加速倍数
    success_count = 0#成功次数的计数
    len_origin_astar_paths=[]#原始的astar算法路径长度
    len_optimized_paths_w_1=[]
    optimized_times_w_1_05=[]
    len_optimized_paths_w_1_05=[]
    optimized_times_w_1_1=[]#这个是设置权重为1.1的双重加速的版本
    len_optimized_paths_w_1_1=[]#这个是权重为1.1的路径长度记录
    optimized_times_w_1_3=[]
    len_optimized_paths_w_1_3=[]
    optimized_times_w_1_5=[]
    len_optimized_paths_w_1_5=[]
    optimized_times_w_2_0=[]
    len_optimized_paths_w_2_0=[]
    for i in range(num_tests):
        os.makedirs(save_root+f"{i+1}_out",exist_ok=True)
        space_map = create_3d_obstacles()#space = np.ones((MAX_X, MAX_Y, MAX_Z))，随机在三维空间中生成立方体的障碍物
        start_pt, target_pt = get_random_points(space_map, min_dist=60)#随机在space map中选择起点和终点距离超过min_dist的两点
        print(f"found_pt: {start_pt}, target_pt: {target_pt}")#打印起点和终点
        
        # 测试两次加速版本，指的是用上了全部的加速方法的版本,设置function_h_w=1.2
        t0 = time.time()
        found_opt,path1 = astar_search_3d_optimized_w(start_pt, target_pt, space_map,function_h_w=1.2)
        path_0=path1
        t_opt = time.time() - t0
        print(f"双次优化的耗费时间(w=1.2)：{t_opt:.4f}")

        #测试两次加速版本，设置function_h_w=1.05
        t_1_05=time.time()
        found_opt_1_05,path_1_05=astar_search_3d_optimized_w(start_pt, target_pt, space_map,function_h_w=1.05)
        t_1_05_opt=time.time()-t_1_05
        print(f"双次优化的耗费时间(w=1.05)：{t_1_05_opt:.4f}")

        #测试两次加速版本，设置function_h_w=1.1
        t_1_1=time.time()
        found_opt_1_1,path_1_1=astar_search_3d_optimized_w(start_pt, target_pt, space_map,function_h_w=1.1)
        t_1_1_opt=time.time()-t_1_1
        print(f"双次优化的耗费时间(w=1.1)：{t_1_1_opt:.4f}")

        #测试两次加速版本，设置function_h_w=1.3
        t_1_3=time.time()
        found_opt_1_3,path_1_3=astar_search_3d_optimized_w(start_pt, target_pt, space_map,function_h_w=1.3)
        t_1_3_opt=time.time()-t_1_3
        print(f"双重优化的耗费时间(w=1.3):{t_1_3_opt:.4f}")

        #测试两次加速版本，设置function_h_w=1.5
        t_1_5=time.time()
        found_opt_1_5,path_1_5=astar_search_3d_optimized_w(start_pt,target_pt,space_map,function_h_w=1.5)
        t_1_5_opt=time.time()-t_1_5
        print(f"双次优化的耗费时间(w=1.5):{t_1_5_opt:.4f}")


        #测试两次加速版本，设置function_h_w=2.0
        t_2_0=time.time()
        found_opt_2_0,path_2_0=astar_search_3d_optimized_w(start_pt,target_pt,space_map,function_h_w=2.0)
        t_2_0_opt=time.time()-t_2_0
        print(f"双次优化的耗费时间(w=2.0):{t_2_0_opt:.4f}")

        trajectory_1=time.time()
        path1=b_spline_optimization(path1)#轨迹优化，轨迹优化不一定会用到
        path1=list(path1)
        trajectory_cost=time.time()-trajectory_1
        #print(f"轨迹优化时间: {trajectory_cost}")
        visualize_3d_path_save(start_pt,target_pt,path1,space_map,save_path=save_root+f"{i+1}_out\\{i+1}_acc_b.png")#b是指的是b spline轨迹优化
        
        visualize_3d_path_save(start_pt,target_pt,path_0,space_map,save_path=save_root+f"{i+1}_out\\{i+1}_acc.png")
        #测试仅仅使用了数据结构的优化的版本，主要是heapq的引入
        t_data_structure_begin=time.time()

        found_data_structure,path_data_structure=astar_search_3d_with_data_structure_optimization(start_pt,target_pt,space_map=space_map)#测试仅仅使用了数据结构优化的结果
        t_data_structure_end=time.time()
        t_data_structure_duration=t_data_structure_end-t_data_structure_begin
        data_structure_optimization_times.append(t_data_structure_duration)
        print(f"仅仅使用数据结构优化的耗时：{t_data_structure_duration:.4f}")
        #测试仅仅计算了相应的邻近值的版本

        t_precompute_neighbors_begin=time.time()
        found_precompute_neighbors,path_precompute_neighbors=astar_search_3d_with_precomputed_neighbors(start_pt,target_pt,space_map=space_map)#测试仅仅使用了预先计算邻近值的结果
        t_precompute_neighbors_end=time.time()
        t_precompute_duration=t_precompute_neighbors_end-t_precompute_neighbors_begin
        #t_precompute_duration=0.01
        print(f"仅仅使用了邻近值预先计算的耗费时间：{t_precompute_duration:.4f}")
    
        # 测试单次加速版本，单次加速指的是没有用加权启发函数的版本
        t2=time.time()
        found_orig,path2 = astar_search_3d_optimized(start_pt, target_pt, space_map)
        t_single = time.time() - t2
        print(f"没有使用加权启发函数版本的耗费时间：{t_single:.4f}")
        visualize_3d_path_save(start_pt,target_pt,path2,space_map,save_path=save_root+f'{i+1}_out\\{i+1}_easy_acc.png')
        # 测试原始版
        t1 = time.time()
        found_non,path3=astar_search_3d(start_pt, target_pt, space_map)
        path3=path2
        t_orig = time.time() - t1
        #t_orig=1.0
        print(f"原始的耗费时间：{t_orig:.4f}")
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
        len_acc=get_paths_total_length(path_0)#双重加速后的路径长度
        len_easy_acc=get_paths_total_length(path2)#单次加速（不包含加权启发函数）后的路径长度
        len_origin=get_paths_total_length(path3)#原始的A star路径规划的轨迹
        len_rrt=get_paths_total_length(path4)
        len_rrt_origin=get_paths_total_length(path5)#获得RRT origin的路径长度
        len_origin_astar_paths.append(len_origin)#原始的A star路径规划的轨迹长度
        len_optimized_w_1_05=get_paths_total_length(path_1_05)
        len_optimized_w_1_1=get_paths_total_length(path_1_1)
        len_optimized_w_1_3=get_paths_total_length(path_1_3)
        len_optimized_w_1_5=get_paths_total_length(path_1_5)
        len_optimized_w_2_0=get_paths_total_length(path_2_0)
        len_optimized_paths_w_1_05.append(len_optimized_w_1_05)
        len_optimized_paths_w_1.append(len_easy_acc)

        len_optimized_paths_w_1_1.append(len_optimized_w_1_1)
        len_optimized_paths_w_1_3.append(len_optimized_w_1_3)
        len_optimized_paths_w_1_5.append(len_optimized_w_1_5)
        len_optimized_paths_w_2_0.append(len_optimized_w_2_0)

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
            optimized_times_w_1_05.append(t_1_05)
            optimized_times_w_1_1.append(t_1_1)#append上时间
            optimized_times_w_1_3.append(t_1_3)
            optimized_times_w_1_5.append(t_1_5)
            optimized_times_w_2_0.append(t_2_0)

            single_optimized_times.append(t_single)
            precompue_neighbor_times.append(t_precompute_duration)
            data_structure_optimization_times.append(t_data_structure_duration)
            data_structure_optimization_speedups.append(t_orig/t_data_structure_duration)
            precompute_neighbors_speedups.append(t_orig/t_precompute_duration)
            speedups.append(t_orig/t_opt)
            single_speedups.append(t_orig/t_single)
            rrt_related_speedups.append(cost_rrt/t_opt)
            rrt_origin_related_speedups.append(cost_rrt_origin/t_opt)
            print(f"测试 {i+1:02d}: 原始 {t_orig:.4f}s |单次优化{t_single:.4f}s| 二次优化 {t_opt:.4f}s | 二次加速 {t_orig/t_opt:.2f}x|单次加速{t_orig/t_single:.2f}x")
        else:
            print(f"测试 {i+1:02d}: 路径不可达，跳过数据统计")

    if success_count > 0:
        avg_orig = np.mean(original_times)#原始版本 
        avg_opt = np.mean(optimized_times)#双重加速的
        avg_single_opt = np.mean(single_optimized_times)
        avg_rrt=np.mean(rrt_times)
        avg_rrt_origin=np.mean(rrt_origin_times)
        avg_precompute_neighbors_speedups=np.mean(precompute_neighbors_speedups)
        avg_precompute_neighbors_cost=np.mean(precompue_neighbor_times)

        avg_data_structure_optimization_cost=np.mean(data_structure_optimization_times)
        avg_data_structure_optimization_speedups=np.mean(data_structure_optimization_speedups)

        avg_len_optimized_paths=np.mean(len_optimized_paths)
        avg_len_rrt_star=np.mean(len_rrt_star)
        avg_len_rrt_origin=np.mean(len_rrt_origin_paths)

        avg_len_origin_astar_paths=np.mean(len_origin_astar_paths)#原始的A star路径规划的轨迹长度

        print("\n" + "="*30)
        print(f"测试完成！成功找到路径次数: {success_count}")
        print(f"原始算法平均耗时: {avg_orig:.4f} s")
        print(f"优化算法平均耗时: {avg_opt:.4f} s")
        print(f"平均二次加速比: {avg_orig/avg_opt:.2f} 倍")
        print(f"平均一次加速比:{avg_orig/avg_single_opt:.2f}倍")#所谓一次加速比也就是没有使用启发函数加速的一种
        print(f"平均RRT*耗时:{avg_rrt:.4f}")
        print(f"平均RRT origin耗时:{avg_rrt_origin:.4f}")
        print(f"对比于RRT*的加速比:{avg_rrt/avg_opt:.2f}")
        print(f"对比于RRT的加速比:{avg_rrt_origin/avg_opt:.2f}")
        print(f"average_len_optimized_paths:{avg_len_optimized_paths:.4f}")
        print(f"平均的预先计算邻近值的加速比:{avg_precompute_neighbors_speedups:.4f}")
        print(f"平均的预先计算邻近值的耗时:{avg_precompute_neighbors_cost:.4f}")
        print(f"平均的数据结构优化的耗时:{avg_data_structure_optimization_cost:.4f}")
        print(f"平均的数据结构优化的加速比:{avg_data_structure_optimization_speedups:.4f}")

        print("="*30)
        print("下面是规划的路径长度相关的结果：")
        print(f"average_len_rrt_star:{avg_len_rrt_star:.4f}")
        print(f"average_len_rrt_origin:{avg_len_rrt_origin:.4f}")
        print(f"路径缩减比例(相对于RRT*)：{avg_len_rrt_star/avg_len_optimized_paths:.4f}")
        print(f"路径缩减比例(相对于RRT origin)：{avg_len_rrt_origin/avg_len_optimized_paths:.4f}")
        print(f"average_len_optimized_paths:{avg_len_optimized_paths:.4f}")
        print(f"原始A star路径的平均长度：{avg_len_origin_astar_paths:.4f}")

        print("="*30)
        print("下面是对于w的消融实验的结果")
        print(f"average_len_optimized_paths_w_1:{np.mean(len_optimized_paths_w_1):.4f}")
        print(f"average_len_optimized_paths_w_1_05:{np.mean(len_optimized_paths_w_1_05):.4f}")
        print(f"average_len_optimized_paths_w_1_1:{np.mean(len_optimized_paths_w_1_1):.4f}")
        print(f"average_len_optimized_paths_w_1_2:{avg_len_optimized_paths:.4f}")
        print(f"average_len_optimized_paths_w_1_3:{np.mean(len_optimized_paths_w_1_3):.4f}")
        print(f"average_len_optimized_paths_w_1_5:{np.mean(len_optimized_paths_w_1_5):.4f}")
        print(f"average_len_optimized_paths_w_2_0:{np.mean(len_optimized_paths_w_2_0):.4f}")
        print("下面是时间上的对比")
        def cacl_avg(list1,num_tests:10):
            total=0.0
            for _ in list1:
                total+=_
            return total/num_tests
        print(f"average_optimized_times_w_1:{avg_single_opt:.4f}")
        print(f"average_optimized_times_w_1_05:{cacl_avg(optimized_times_w_1_05,num_tests=num_tests):.4f}")
        print(f"average_optimized_times_w_1_1:{cacl_avg(optimized_times_w_1_1,num_tests=num_tests):.4f}")
        print(f"average_optimized_times_w_1_2:{avg_opt:.4f}")
        print(f"average_optimized_times_w_1_3:{cacl_avg(optimized_times_w_1_3,num_tests=num_tests):.4f}")
        print(f"average_optimized_times_w_1_5:{cacl_avg(optimized_times_w_1_5,num_tests=num_tests):.4f}")
        print(f"average_optimized_times_w_2_0:{cacl_avg(optimized_times_w_2_0,num_tests=num_tests):.4f}")


        
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