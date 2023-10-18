# 基本概念

* **满二叉树** ：深度为 $k$ ，结点数 $2^k-1$ 的二叉树
* **完全二叉树** ：只有最底层没有填满，并且最底层的结点都在左边
* **二叉搜索树** ：结点是有顺序的
* **平衡二叉搜索树** ：左右两子树的高度差不超过 $1$

# 遍历方法

## 深度遍历

实现方法：递归法比较简单就略过了，考虑迭代法的实现。

* 前序：中左右
* 后序：左右中（后序和前序遍历写法是一样的，因为可以理解成在前序过程中，将左右指针交换，最后把结果反转得到的：中右左 -> 左右中）
* 中序：左中右

迭代法的实现首先需要一个新建一个单独的**指针**用来访问树上的结点，并且使用一个临时的**栈**来辅助遍历。以前序遍历为例，实现如下：

```python
class Solution:
    def preorderTraversal(self, root: TreeNode) -> List[int]:
        res = []
        if not root:
            return res
  
        stack = []
        node = root
        while stack or node:
            while node:
                res.append(node.val)
                stack.append(node)
                node = node.left
            node = stack.pop()
            node = node.right
        return res
```

## 广度遍历

广度遍历的迭代法需要一个**队列**来辅助遍历：

```python
class Solution:
    def levelOrder(self, root: Optional[TreeNode]) -> List[List[int]]:
        if not root: return []
        res, queue = [], collections.deque()
        queue.append(root)
        while queue:
            tmp = []
            for _ in range(len(queue)):
                node = queue.popleft()
                tmp.append(node.val)
                if node.left: 
			        queue.append(node.left)
                if node.right: 
			        queue.append(node.right)
                    res.append(tmp)
        return res
```

# 参考习题

| 题目                                                                                                             | 提示                           |
| ---------------------------------------------------------------------------------------------------------------- | ------------------------------ |
| [填充每个节点的下一个右侧节点指针 II](https://leetcode.cn/problems/populating-next-right-pointers-in-each-node-ii/) | 考虑只使用常数空间应该如何求解 |
|                                                                                                                  |                                |
