| Class         | Description        | Method                                                                                 |
| ------------- | ------------------ | -------------------------------------------------------------------------------------- |
| List          | 新建列表           | List `<Integer>` a = new ArrayList<>()                                               |
|               | 加入元素           | a.add(1);                                                                              |
|               | 移除元素           | a.remove(4);                                                                           |
|               | 对列表排序         | Collections.sort(a);                                                                   |
|               | 将列表反转         | Collections.reverse(a);                                                                |
|               | 统计列表中元素     | Collections.max(a);<br />Collections.min(a);                                           |
| Set           | 新建集合           | Set `<Character>` s = new HashSet `<Character>`();                                 |
|               | 加入元素           | s.add('a');                                                                            |
|               | 移除元素           | s.remove('a');                                                                         |
|               | 查询是否包含某元素 | s.contains('a');                                                                       |
|               | 清除集合           | s.clear();                                                                             |
| Map           | 新建哈希表         | Map<Integer,Integer> m = new HashMap<>();                                              |
|               | 加入新元素         | m.put(a, b);                                                                           |
|               | 获取元素           | m.get(k);                                                                              |
|               | 查询元素           | m.containsKey(k)                                                                       |
| Queue         | 新建队列           | Queue `<Integer>` q = new LinkedList<>();                                            |
|               | 加入元素           | q.offer(1);                                                                            |
|               | 弹出元素           | q.poll();                                                                              |
| Deque         | 新建双向队列       | Deque `<Integer>` dq = new LinkedList<>();                                           |
|               | 加入元素           | dq.addFirst(1);<br />dq.addLast(1);                                                    |
|               | 弹出元素           | dq.pollFirst();<br />dq.pollLast();                                                    |
|               | 获得元素           | dq.peekFirst();<br />dq.peekLast();                                                    |
| PriorityQueue | 新建优先队列       | PriorityQueue `<Integer>` pq = new PriorityQueue<>(((o1, o2) -> {return o2 - o1;})); |
