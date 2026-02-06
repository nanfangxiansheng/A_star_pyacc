

# 加速版本A*算法的python实现

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



## 加速版本的A*算法的python实现

加速版本的第一个改进方案即减少distance计算次数：通过预先对一个点周围的26个点计算偏移向量和距离来节省计算消耗。除此之外，还引入了heapq结构，在python中heap q结构内部维护了一个二叉树，作为一个堆，其在每次内部寻找的时候不需要线性查表，只需要pop值即可。

在A*算法的实现过程中，之前的方法需要频繁的查重和更新（如果g值更小则进行更新）。而现在的改进方法引入了一个g score字典：{(x,y,z):cost}。由于Python中字典的本质是映射的哈希表，所以根据键来查找值几乎是瞬间完成的，可以节省大量时间。



## 测试加速倍数的基准

为了测试加速倍数，首先需要确定测试次数，这里设置了测试30次取平均值，在每次测试中，开始点和结束点都是随机设置的，但保证两者间的距离大于某个阈值，在三维空间中随机生成一批障碍物，大小均是随机的，在每次测试中加速和没加速在方法在同样的一张地图中进行，且开始点和结束点都是一样的。

![Figure_1](.\figure\Figure_1.png)

最终加速方法和原始方法的对比结果如上所示，可以看到在应用了加速方法后测试耗时大幅度缩减且耗时维持在稳定的范围中。

而寻找路径的可视化结果如下所示，深灰色的即障碍物，中间的线即轨迹。

![5_acc](.\figure\4_acc.png)