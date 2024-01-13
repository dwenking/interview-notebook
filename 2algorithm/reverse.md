# 基本概念

回溯法是一种搜索算法，使用递归实现。回溯法的效率并不高，因此只能在数据量小的情况下使用。

回溯法，一般可以解决如下几种问题：

* 组合问题：N个数里面按一定规则找出k个数的集合
* 切割问题：一个字符串按一定规则有几种切割方式
* 子集问题：一个N个数的集合里有多少符合条件的子集
* 排列问题：N个数按一定规则全排列，有几种排列方式
* 棋盘问题：N皇后，解数独等等

回溯算法注意事项：

* 终止条件：注意算法一定要写好终止条件，否则无法终结

# 排列与组合

什么时候使用 `used` 数组，什么时候使用 `beginIdx` 变量：

* 排列问题，讲究顺序（即 [2, 2, 3] 与 [2, 3, 2] 视为不同列表时），需要记录哪些数字已经使用过，此时用 `used` 数组；
* 组合问题，不讲究顺序（即 [2, 2, 3] 与 [2, 3, 2] 视为相同列表时），需要按照某种顺序搜索，此时使用 `beginIdx` 变量（因为没有顺序，因此最开始就要定下一个顺序）。

排列：

```java
class Solution {
    LinkedList<Integer> tmp = new LinkedList<>();
    List<List<Integer>> res = new ArrayList<>();

    public List<List<Integer>> permute(int[] nums) {
        boolean[] use = new boolean[nums.length];
        backTracking(nums, use);
        return res;
    }

    private void backTracking(int nums[], boolean[] use) {
        if (tmp.size() == nums.length) {
            res.add(new ArrayList<Integer>(tmp));
            return;
        }

        for (int i = 0; i < nums.length; i++) {
            if (use[i]) continue;
            use[i] = true;
            tmp.add(nums[i]);
            backTracking(nums, use);
            tmp.removeLast();
            use[i] = false;
        }
    }
}
```

组合：

```java
class Solution {
    List<List<Integer>> res = new ArrayList<>(); 
    LinkedList<Integer> list = new LinkedList<>();

    public List<List<Integer>> subsets(int[] nums) {
        backTracking(nums, 0);
        return res;
    }

    public void backTracking(int[] nums, int idx) {
        res.add(new ArrayList<Integer>(list));
        int s = nums.length;

        for (int i = idx;i < s;i++) {
            list.add(nums[i]);
            backTracking(nums, i+1);
            list.removeLast();
        }
    }
}
```

# 参考习题

| 题目 | 提示 |
| ---- | ---- |
|      |      |
|      |      |
