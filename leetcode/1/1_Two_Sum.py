# coding=utf-8

class Solution(object):
    """docstring for Solution"""
    def twoSum(self, nums, target):
        for i in range(len(nums)):  #这里要用长度！！！！！
            for j in range(i+1,len(nums)):    #这里主要要加1，否则还有自身！！！！！！
                print(i,j)
                if nums[i]+nums[j] == target:   # 这里要用真实值相乘
                    return [i,j]    # 返回值是list不是tuple
                else:
                    continue

if __name__ == '__main__':
    nums = [1, 3, 4, 7]
    target = 7
    a = Solution()
    result = a.twoSum(nums, target)
    print(result)
        