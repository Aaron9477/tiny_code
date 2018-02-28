class Solution(object):
    def removeDuplicates(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        nums.sort() # 先排序，再按顺序进行剔除
        i = 0
        while(i < len(nums)-1):    #这里用小于等于，防止溢出
            while(nums[i]==nums[i+1]):  # del之后，索引不需要变，后面的数字会成为刚删掉的索引的位置
                del nums[i+1]
                if(i == len(nums)-1):
                    break   # 之前在这里直接return不可以，因为最后两个数字不一样不会进入这个while，会出错
            i += 1
        return len(nums)