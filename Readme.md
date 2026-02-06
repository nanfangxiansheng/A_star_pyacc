

# 加速版本A*算法的python实现

适用于无人机在三维场景下的路径规划A*算法的探索和优化

## 原始版本三维A*算法的python实现

A*算法通过维护两个列表来进行搜寻最短路径：

一个是**开放列表**，一个是关闭列表。开放列表 存储了所有已经被发现，但尚未被完全探索的节点。可以把它想象成“待考察”的节点集合。通常使用优先队列（Min-Heap）实现，以便快速找到最有希望的节点。

而在**关闭列表**中， 存储了所有已经被访问和探索过的节点。防止重复处理。通常使用哈希集合（Set）实现，以便快速检查节点是否已被访问。

对于每个节点n,A*算法会计算一个评估函数f(n),如下所示：
$$
f(n)=g(n)+h(n)
$$
在其中：

- `g(n)`：从**起始节点**到节点 `n` 的**实际移动成本**。这是已知的、精确的值。
- `h(n)`：从节点 `n` 到**目标节点**的**启发式估计成本**（Heuristic）。这是一个基于某些规则（如直线距离）的猜测值，用于指导搜索方向。启发函数的设计至关重要，它需要是**可接受的（Admissible）**，即永远不会高估实际成本。
- `f(n)`：节点 `n` 的综合优先级。A* 算法总是优先探索 `f(n)` 值最小的节点。

对于三维空间中，h(n)一般采用的是欧几里得距离，也就是三维空间两点的距离。下面的代码中展示了这一过程：

首先将起始节点放在open list,当 Open List 不为空时：

 a. 从 Open List 中取出 `f(n)` 值最小的节点 `current`。

 b. 如果 `current` 是目标节点，则成功找到路径，回溯构造路径并结束。

 c. 将 `current` 从 Open List 移除，并放入 Closed List。

 d. 遍历 `current` 的所有**邻居节点**`neighbor`：

 i. 如果 `neighbor` 在 Closed List 中，或者 `neighbor` 是障碍物，则忽略。

 ii. 计算从 `start` 经过 `current` 到达 `neighbor` 的 `g` 值 (`tentative_g`)。

 iii. 如果 `neighbor` 不在 Open List 中，或者 `tentative_g` 小于 `neighbor` 当前记录的 `g` 值： - 更新 `neighbor` 的父节点为 `current`。 - 更新 `neighbor` 的 `g` 值 (`g(neighbor) = tentative_g`)。 - 计算 `neighbor` 的 `h` 值 (`h(neighbor)`)。 - 计算 `neighbor` 的 `f` 值 (`f(neighbor) = g(neighbor) + h(neighbor)`)。 - 如果 `neighbor` 不在 Open List 中，将其加入 Open List。

d中关键部分的代码实践如下所示：

```python
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

```



## 基于数据结构计算优化的单次加速版本的A*算法的python实现

加速版本的第一个改进方案即减少distance计算次数：通过预先对一个点周围的26个点计算偏移向量和距离来节省计算消耗。除此之外，还引入了heapq结构，在python中heap q结构内部维护了一个二叉树，作为一个堆，其在每次内部寻找的时候不需要线性查表，只需要pop值即可。

在A*算法的实现过程中，之前的方法需要频繁的查重和更新（如果g值更小则进行更新）。而现在的改进方法引入了一个g score字典：{(x,y,z):cost}。由于Python中字典的本质是映射的哈希表，所以根据键来查找值几乎是瞬间完成的，可以节省大量时间。

## 基于加权启发函数的二次加速的A*算法的python实现

在原来的版本中，计算启发函数时候严格依靠
$$
f(n)=g(n)+h(n)
$$
其中g(n)即累积值，衡量无人机在该点已经移动了多远，h(n)是预计值，衡量无人机在该点到目标还有多少距离。因此为了使得无人机在路径规划的过程中更加具有方向性，考虑给h(n)乘上一个权重系数w,则上面的公式变为了：
$$
f(n)=g(n)+w*h(n)
$$
这样使得其在f(n)相同的时候，更加优先选择h(n)小的邻近点。实践中给w的值设定为1.2并且基于此实验，发现加速效果大幅度的提升，在50\*50\*50格的带有随机障碍物三维空间中较长距离路径规划时间花费达到了ms甚至于低于ms的级别。

## 测试加速倍数和其效果

为了测试加速倍数，首先需要确定测试次数，这里设置了测试50次取平均值，在每次测试中，开始点和结束点都是随机设置的，但保证两者间的距离大于某个阈值，在三维空间中随机生成一批障碍物，大小均是随机的，在每次测试中加速和没加速在方法在同样的一张地图中进行，且开始点和结束点都是一样的。

![](.\figure\Figure_1.png)

经过50次测试发现最终的采用二次加速后的平均路径规划时间花费为0.0063s。而原始的路径规划平均时间花费为5.707s，采用了二次加速的平均加速倍数高达904倍，而采用了单次数据结构优化加速的倍数达到了45倍。

这里定义平均加速倍数为:
$$
\frac{T_{original,mean}}{T_{acc,mean}}
$$
证明了方法的有效性。最终加速方法和原始方法的对比结果如上所示，可以看到在应用了加速方法后测试耗时大幅度缩减且耗时维持在稳定的范围中。

## 路径轨迹优化的探索

而寻找路径的可视化结果如下所示，深灰色的即障碍物，中间的线即轨迹。

![5_acc](.\figure\4_acc.png)

在路径轨迹优化方面，由于无人机在空中飞行的时候，期望的轨迹应当是平滑的，对于折角过大的轨迹是不能接受的，因此在这里引入了一种基于b样条插值的三维轨迹优化策略，相比于简单的路径修剪，B样条可以生成**连续、高阶可导**的曲线，这对于无人机或机器人的电机控制至关重要（因为它保证了速度和加速度的平滑）。

其实现代码逻辑如下：

```python
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
```

采用了B样条轨迹优化前后的对比结果如下所示：

![](.\figure\1_acc.png)

优化后的结果为：

![](.\figure\1_acc_b.png)

可以看到采用了基于b样条插值后的优化方案后，轨迹更加平滑了，而且轨迹优化的时间花费较为固定，均在1ms级别。在定量分析中，经过五十次实验，发现经过轨迹优化后的路径长度平均值较优化前减少了2.3%.



## 更加广泛的比较

为了更加全面的衡量改进后的A*算法的性能，在这里接入了RRT算法及其的变体来作为比较。 **RRT**（Rapidly-exploring Random Tree）快速扩展随机树是一种采样式路径规划算法，广泛应用于机器人运动规划、自动驾驶、无人机路径设计等领域。它特别适用于高维空间中的路径规划问题。 RRT的核心思想是通过在空间中随机采样点并逐步构建一棵树形结构（搜索树），来快速探索空间并找到从起点到终点的可行路径。 RRT偏向于快速探索未被探索的空间区域，从而快速覆盖整个搜索空间。

RRT算法的基本流程如下：

步骤：
1.初始化一棵树 T，树的根节点为起点 q_start。
2.对于每次迭代：

- 随机采样一个点 q_rand（可以是完全随机，也可以有一定概率采样为 q_goal，称为“目标偏向”）。-
- 在树中找到距离 q_rand 最近的节点 q_nearest。
- 从 q_nearest 向 q_rand 移动一个固定步长 Δq，得到新的节点 q_new。
- 如果 q_new 不在障碍物中，则将其加入树中，并将其父节点设为 q_nearest。
- 如果 q_new 距离 q_goal 很近，可以认为找到了可行路径。

3.如果找到路径，沿父节点回溯得到路径；否则直到达到最大迭代次数。

而RRT*算法作为RRT算法的改进版本，其加入了“路径优化”的机制：在每次加入新节点时，不仅连接最近点，还会尝试重新连接周围节点，以获得更短路径。这使得其在理论上可以获得渐进的最优解。

RRT star算法的三维空间可视化如下所示,可以看到在三维空间中从起点到终点有数条错误的蓝色路径，而最终找到了可以规避所有障碍物的红色路径。

![](.\figure\rrt.png)

经过二十次随机测试，通过在相同的仿真空间中对比采用了两重加速和轨迹优化的A\*算法和RRT\*算法的性能如下：

![](.\figure\compare_rrtstar.png)

计算得到优化后的A\*算法对比于RRT\*算法的加速比平均值为：5.85倍。而路径长度缩减幅度达到了15%。

而在和RRT算法的对比过程中，经过二十次随机测试，在相同的仿真空间中对比采用了两重加速和轨迹优化的A\*算法和RRT\*算法的性能如下：

![](.\figure\RRT_compare.png)

计算得到优化后的A*算法对比于RRT算法的加速比平均值为3.10倍，而路径长度的缩减平均值达到了37%。而这样的结果也与RRT\*和RRT之间的比较相符。RRT\*虽然缩短了轨迹长度，但是使得计算复杂度增加。
