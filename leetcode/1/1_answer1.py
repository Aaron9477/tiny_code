# coding=utf-8
# 利用字典效率会高很多
class Solution(object):
    """docstring for solution"""
    def twoSum(self, nums, target):
        if len(nums)<1:
            return False
        buff_dict = {}  # 建立字典
        for i in range(len(nums)):
            if nums[i] in buff_dict:    # 如果该值的另一半储存在字典中就返回
                return [buff_dict[nums[i]],i]
            else:
                buff_dict[target-nums[i]] = i   # 如果该值不在字典中，就存储

if __name__ == '__main__':
    nums = [1, 3, 4, 7]
    target = 7
    a = Solution()
    result = a.twoSum(nums, target)
    print(result)
