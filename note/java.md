# Java

| Class          | Description        | Method                                                                                             |
| -------------- | ------------------ | -------------------------------------------------------------------------------------------------- |
| 控制台输入输出 | 创建对象           | Scanner input = new Scanner(System.in);                                                        |
|                | 读取整数           | int a = input.nextInt();                                                                           |
|                | 读取空格分割字符   | String s = input.next();                                                                           |
|                | 读取回车分割字符   | String s = input.nextLine();                                                                       |
|                | 循环读取           | while (input.hasNextLine(s))                                                                       |
|                | 格式化输出         | System.out.format();                                                                               |
| Arrays         | 排序               | Arrays.sort(a);                                                                                    |
|                | 填充               | Arrays.fill(a, 1);                                                                                 |
| List           | 新建列表           | List `<Integer>` a = new ArrayList<>();                                                          |
|                | 加入元素           | a.add(1);                                                                                          |
|                | 移除元素           | a.remove(4);                                                                                       |
|                | 对列表排序         | Collections.sort(a);                                                                               |
|                | 将列表反转         | Collections.reverse(a);                                                                            |
|                | 统计列表中元素     | Collections.max(a);<br />Collections.min(a);                                                       |
| Set            | 新建集合           | Set `<Character>` s = new HashSet `<Character>`();                                             |
|                | 加入元素           | s.add('a');                                                                                        |
|                | 移除元素           | s.remove('a');                                                                                     |
|                | 查询是否包含某元素 | s.contains('a');                                                                                   |
|                | 清除集合           | s.clear();                                                                                         |
| Map            | 新建哈希表         | Map<Integer,Integer> m = new HashMap<>();                                                          |
|                | 加入新元素         | m.put(a, b);                                                                                       |
|                | 获取元素           | m.get(k);                                                                                          |
|                | 查询元素           | m.containsKey(k)                                                                                   |
|                | 元素不存在时的处理 | m.putIfAbsent(k, 0);<br />m.getOrDefault(k, 0);                                                    |
|                | 遍历元素           | for (Map.Entry<String, Integer> en : m.entrySet());<br />en.getKey();<br />en.getValue();          |
| Stack          | 新建栈             | Stack `<Integer>` s = new LinkedList<>();                                                        |
|                | 加入元素           | s.push(1);                                                                                         |
|                | 弹出元素           | s.pop();                                                                                           |
| Queue          | 新建队列           | Queue `<Integer>` q = new LinkedList<>();                                                        |
|                | 加入元素           | q.offer(1);                                                                                        |
|                | 弹出元素           | q.poll();                                                                                          |
| Deque          | 新建双向队列       | Deque `<Integer>` dq = new LinkedList<>();                                                       |
|                | 加入元素           | dq.addFirst(1);<br />dq.addLast(1);                                                                |
|                | 弹出元素           | dq.pollFirst();<br />dq.pollLast();                                                                |
|                | 获得元素           | dq.peekFirst();<br />dq.peekLast();                                                                |
| PriorityQueue  | 新建优先队列       | PriorityQueue `<Integer>` pq = new PriorityQueue<>(((o1, o2) -> {return o2 - o1;})); // 默认小根 |
|                | 查询堆顶           | pq.peek();                                                                                         |
|                | 加入元素           | pq.offer(1);                                                                                       |
|                | 弹出元素           | pq.poll();                                                                                         |
