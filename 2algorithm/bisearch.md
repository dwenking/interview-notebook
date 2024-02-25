# 基本概念

二分查找一般用于查找有序数组中的元素。除此之外，还能够处理单调值问题比如最大值最小。

注意事项：

* 区间的选择（左闭右开/左闭右闭）和停止条件、转移方程要对应起来
* 所有二分查找都要注意**缩小时的边界问题**，是否会死循环（可以模拟**数组里只有两个数**的情况，看是否会死循环）

# 模版

## = target

注意，因为选择的是左闭右开的区间里，也就是 [left, right)，那么题中：

* while (left < right)，这里使用 < ,因为left == right在区间 [left, right) 是没有意义的
* right 更新为 mid，因为当前 nums[mid] 不等于 target ，去左区间继续寻找，而寻找区间是左闭右开区间，所以right更新为middle

```java
public int search(int[] nums, int target) {
    int left = 0, right = nums.length;
    while (left < right) {
        int mid = (left + right) / 2;
        if (nums[mid] == target) {
            return mid;
        } else if (nums[mid] > target) {
            right = mid;
        } else {
            left = mid + 1;
        }
    }
    return -1;
}
```

# 参考习题

| 题目                                                                                     | 提示                                                                         |
| ---------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------- |
| [分割数组的最大值](https://leetcode.cn/problems/split-array-largest-sum/)                   | 二分查找的最大值最小问题，已知要将数组分成m部分，二分m部分的最大值           |
| [标记所有下标的最早秒数 I](https://leetcode.cn/problems/earliest-second-to-mark-indices-i/) | 时间越大，越能够标记所有下标，因此具有**单调性**。使用二分能够解决问题 |
