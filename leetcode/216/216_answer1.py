# using coding=utf-8
# 本题使用了递归的思路
class Solution(object):
    """docstring for Solution"""
    def combinationSum3(self, k, n):
        if n > sum([i for i in range(1,10)]):   # 最大只能达到55，超出直接退出
            return []
        self.res = []   # 初始集合
        self.help_sum(k,n,1,[],self.res)    # 这里不需要定义一个list，直接输入[]即可
        return self.res

    def help_sum(self, k, n, curr, arr, res):
        if len(arr) == k:   # 如果到达数量了，就停止
            if sum(arr) == n:   # 到达数量的同时，判断是否满足和的要求
                print(arr)
                self.res.append(list(arr))  # 满足将结果加入到输出数组里 # 这里一定要加list！！！！！！不加list结果错误！！！！！
            return  # return可以直接用，后面不用加参数

        if len(arr) > k or curr > 9:    # 退出条件，这两句话不加也可以
            return

        for i in range(curr,10):    # 注意，这里有个循环，而且是从curr开始的循环，比它小的数已经试过了
            arr.append(i)   # 将当前数放到数组里
            # [1,2,4]怎么得到的，当[1,2,3,4]输入之后会返回，pop出来，尝试5-9都pop出来，然后会把3pop出来，一试4就成功了。
            self.help_sum(k, n, i+1, arr, self.res) # 进行递归 #注意这里是i+1，不是curr，可能会跳过一些数，比如[1,2,4]跳过了3 
            arr.pop()

if __name__ == '__main__':
    a = Solution()
    k = 2
    n = 9
    print(a.combinationSum3(k,n))
