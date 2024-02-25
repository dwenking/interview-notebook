# 基本概念

图是由**节点**和**边**组成的。图的常用存储方式有两种：

* 邻接表
* 邻接矩阵

**度**：和每个节点相连的边的条数。有向图中的节点具有入度和出度。

# 基本思路

图的题目经常涉及到搜索，在思考过程中需要考虑：

* 终止条件（点满足什么条件、边满足什么条件时终止搜索）

# 遍历方法

## 深度遍历

图的遍历方法和树类似，但一个重要的区别在于，图本身可能包含环。因此为了不重复地遍历每个节点，需要使用$visit$数组进行标记。

以下代码给出遍历图的最基本框架，注意，如果是遍历**无环图**不需要$visit$标记：

```python
def dfs(graph, s):
    if visited[s]:
        return
    visited[s] = True

    for neighbor in graph.neighbors(s):
        dfs(graph, neighbor)
```

## 广度遍历

广度遍历和树一样，需要队列来保存需要搜索的点。由于广度遍历本身是往各个方向搜索的，因此必须$visit$标记点来防止无限循环。

求最短路径常用广搜，当广搜到达终点时，一定是最短路径。

```python
visited =  [[False] * n for _ in range(m)]

def bfs(grid, visited, x, y):
  queue = [] 
  queue.append((x, y)) 
  visited[x][y] = True 
  
  while queue: 
    curx, cury = queue.pop(0)
  
    for dx, dy in (0,1),(1,0),(0,-1),(-1,0):
      nextx, nexty = curx + dx, cury + dy
      if nextx < 0 or nextx >= len(grid) or nexty < 0 or nexty >= len(grid[0]): 
        continue
      if not visited[nextx][nexty]:
        queue.append((nextx, nexty)) 
        visited[nextx][nexty] = True
```

# 并查集

**并查集的主要功能：**

* 将两个元素添加到同一个集合中
* 判断两个元素是否在同一个集合中

**并查集基本元素：**

为了控制代码量，可以考虑只保留并查集的基本元素，其他元素灵活增减。

* `parent`数组，用来表示每个点的根
* `union`、`find` 用于满足并查集基本操作

```python
class UF:
	def __init__(self, n):
		# 初始化时互不连通
		self.count = n
        	# 父节点指针初始指向自己
        	self.parent = [i for i in range(n)]

	def union(self, p, q):
		root_p = self.find(p)
        	root_q = self.find(q)
		if root_p == root_q:
            		return
		self.parent[root_p] = root_q
		self.count -= 1

	def find(self, x: int):
		if self.parent[x] != x:
            		self.parent[x] = self.find(self.parent[x]) # 路径压缩
        	return self.parent[x]

	def connected(self, p: int, q: int) -> bool:
        	root_p = self.find(p)
        	root_q = self.find(q)
        	return root_p == root_q

	def count(self) -> int:
        	return self.count
```

**理解并查集的路径压缩：**

为了压缩并查集中每个树的高度，降低时间复杂度，需要修改 `find`函数。保证任意树的高度保持在常数。路径压缩后的并查集时间复杂度在$O(logn)$与$O(1)$之间，且随着查询或者合并操作的增加，时间复杂度会越来越趋于$O(1)$。

# 最短路径

# 参考习题

| 题目                                                           | 提示                                                                                       |
| -------------------------------------------------------------- | ------------------------------------------------------------------------------------------ |
| [岛屿数量](https://leetcode.cn/problems/number-of-islands/)       | 图论基础题，重要的是从本题掌握DFS、BFS、并查集的技巧                                       |
| [飞地的数量](https://leetcode.cn/problems/number-of-enclaves/)    | 将与边界相连的地都标记成海洋，最后再统计飞地个数                                           |
| [冗余连接](https://leetcode.cn/problems/redundant-connection/)    | 并查集应用，如果在遍历边的过程中，发现了已经处于同一集合的点，则说明存在环                 |
| [最大人工岛](https://leetcode.cn/problems/making-a-large-island/) | 如果使用暴力枚举方法的话复杂度为$O(n^4)$，因此需要做优化，首先统计岛屿面积并保存到数组里 |
| [单词接龙](https://leetcode.cn/problems/word-ladder/)             | BFS统计最短路径                                                                            |
