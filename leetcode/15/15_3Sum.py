#coding=utf-8

class Solution(object):
    def threeSum(self, nums):
        # 这里不能先运行tudyArray，然后判断不同的数字 够不够三个，因为可能有[-1,-1,2]这种形式，不同的数字只有两个，但是能够组成0
        # 三个0的最特殊，必须单独判断，不能在
        if len(nums) < 3:
            return []
        res = []
        arr_set, arr_repeat = self.tidyArray(nums)
        self.helpSumDiff(arr_set, [], 0, res)

    # def tidyArray(self, nums):
    #     nums.sort()
    #     arr_dif = []
    #     arr_repeat = []
    #     for i in nums_tidy:
    #         if i==0 or nums_tidy[i] != arr_dif[-1]:
    #             arr_dif.append(nums_tidy[i])
    #         elif arr_repeat==None or nums_tidy[i] != arr_repeat[-1][-1]:
    #             arr_repeat.append(list([nums_tidy[i], nums_tidy[i]]))
    #     return arr_dif, arr_repeat

    def tidyArray(self, nums):
        arr_set = set(arr)
        arr_repeat = []
        for i in arr_set:
            if nums.count(arr_set[i]) > 1:
                arr_repeat.append(list[arr_set[i], arr_set[i]])
        return arr_set, arr_repeat



    def helpSumDiff(self, nums, arr, curr, res):
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

if __name__ == '__main__':
    a = []
    b = 1
    a.append([b,b])
    print(a)
