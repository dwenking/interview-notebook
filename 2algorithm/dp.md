# 基本概念

动态规划针对的情景是重叠子问题的情景，由前至状态转向下一个状态。与贪心的区别在于，

思考方式：

* 确定DP数组以及下标的含义
* 确定递推公式以及初始化
* 确定遍历顺序

# 背包问题

背包问题是动态规划中一类重要的问题。

$dp[j]$表示：容量为j的背包，所背的物品价值可以**最大**为$dp[j]$，那么$dp[0]$就应该是$0$（背包不一定是装满的，因为只需要价值最大）。

## 01背包

$n$件不同重量和价值的物品，背包的重量为$w$。每件物品只能用**一次**。

注意：代码版本是用滚动数组进行压缩后的版本。因此要注意遍历顺序。**倒序遍历是为了保证物品只被放入一次！** 如果一旦正序遍历了，那么物品就会被重复加入多次。

```java
for(int i = 0; i < weight.size(); i++) { // 遍历物品
    for(int j = bagWeight; j >= weight[i]; j--) { // 遍历背包容量
        dp[j] = max(dp[j], dp[j - weight[i]] + value[i]);
    }
}
```

## 完全背包

和01背包不同的地方在于，每种物品有**无限件**。

```java
for(int i = 0; i < weight.size(); i++) { // 遍历物品
    for(int j = weight[i]; j <= bagWeight ; j++) { // 遍历背包容量
        dp[j] = max(dp[j], dp[j - weight[i]] + value[i]);

    }
}
```

如果求组合数就是外层for循环遍历物品，内层for遍历背包。
如果求排列数就是外层for遍历背包，内层for循环遍历物品。

# 划分DP

需要把数组划分成n个区间。转移方程类似：

```python
dp[i][j] = max(dp[k][j - 1] * xxx, dp[i][j])
```

# 参考习题

| 题目                                                                                                | 提示                                                                                   |
| --------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------- |
| [最小路径和](https://leetcode.cn/problems/minimum-path-sum/)                                           | DP基础题，注意转移方向。如果是单边移动一般是DP，如果各个方向可能是搜索或者图论         |
| [整数拆分](https://leetcode.cn/problems/integer-break/)                                                | 转移方程的书写注意有两种情况：拆分or不拆分                                             |
| [回文子串](https://leetcode.cn/problems/palindromic-substrings/)                                       | 对回文串的判断也存在重叠子问题。重点是考虑DP数组的遍历顺序。根据转移方程来规划遍历顺序 |
| [分割等和子集](https://leetcode.cn/problems/partition-equal-subset-sum/)                               | 01背包第一题，注意本题的value选择有一点绕                                              |
| [零钱兑换](https://leetcode.cn/problems/coin-change/)                                                  | 完全背包第一题，按照模版套公式，注意初始化                                             |
| [零钱兑换 II](https://leetcode.cn/problems/coin-change-ii/)                                            | 完全背包第二题，首先确定求的是组合数，因此DP数组的值代表组合数。还要注意初始化问题     |
| [K 个不相交子数组的最大能量值](https://leetcode.cn/problems/maximum-strength-of-k-disjoint-subarrays/) | 划分DP例题，求解不同数组划分的最大值                                                   |
