# Python

| Class         | Description                    | Method                                                         |
| ------------- | ------------------------------ | -------------------------------------------------------------- |
| 数学运算      | 整除向下取整                   | a // 2                                                         |
|               | 整除向上取整                   | ceil(a / 2)                                                    |
|               | 判断是否整数                   | a.isdigit()                                                    |
|               | 舍入到给定位数                 | round(a, 2)                                                    |
| str           | 统计字符串中的子字符串出现个数 | s.count()                                                      |
|               | list合并成字符串               | "".join(list)                                                  |
| array         | 创建固定大小的数组             | l = [0] * m<br />l = [[0] * m for _ in range(n)]               |
|               | 切片                           | lastTwo = l[-2:]                                               |
| list          | 新建列表                       | l = []                                                         |
|               | 插入新元素                     | l.append(1)<br />l.insert(1, 1)                                |
|               | 移除元素                       | l.remove(1)<br />l.pop(idx)                                    |
|               | 反转列表                       | l.reverse()                                                    |
|               | 对列表进行排序                 | l.sort()<br />l2 = sorted(l, key=lambda x: x[0], reverse=True) |
|               | 统计列表中的元素               | l.count(1)<br />max(l)<br />min(l)                             |
|               | 列表拷贝                       | l2 = l1[:]<br />l2 = copy.deepcopy(l1) # 深拷贝                |
|               | 查找                           | l.index(a)                                                     |
| set           | 新建集合                       | s = set('abracadabra')<br />s = set(l)                         |
|               | 插入新元素                     | s.add('x')                                                     |
|               | 移除元素                       | s.remove('x')                                                  |
| dict          | 新建字典                       | dict = {}                                                      |
|               | 获得元素值                     | dict.get(key, 0)                                               |
|               | 遍历字典                       | for key,value in dict.items():                                 |
| Counter       | 新建计数                       | c = Counter()                                                  |
| stack         | 新建栈                         | s = []                                                        |
|               | 添加元素                       | s.append(1)                                                    |
|               | 弹出元素                       | val = s.pop()                                                  |
| queue         | 新建队列                       | q = []                                                         |
|               | 添加元素                       | q.append(1)                                                    |
|               | 弹出元素                       | q.pop(0)                                                       |
| deque         | 新建双向队列                   | que = collections.deque()                                      |
|               | 添加元素                       | que.append(1)                                                  |
|               | 弹出元素                       | que.popleft()                                                  |
| priorityqueue | 新建优先队列                   | pq = []                                                        |
|               | 插入元素                       | heapq.heappush(pq, 1) # 小根                                   |
|               | 弹出元素                       | heapq.heappop(pq)                                              |
| 其他          | 生成前缀和                     | accumulate(data, initial=0)                                    |
|               | 遍历多个数组                   | for x, f in zip(nums, freq):                                   |
