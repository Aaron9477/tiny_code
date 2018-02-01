#coding=utf-8

class Solution(object):
    def threeSum(self, nums):
        if len(nums) < 3:
            return []
        res = []
        self.helpSum(nums, [], 0, res)


    def helpSum(self, nums, arr, curr, res):
        if len(arr) == 3:
            if sum(arr) == 0:
                res.append(list(arr))
            return

        if len(arr) > 3:
            return

        for i in xrange(curr, len(nums)):
            arr.append(nums[i])
            self.helpSum(nums, arr, i+1, res)
            arr.pop()