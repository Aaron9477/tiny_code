#coding=utf-8
#这个题，坑很多，不容易做
#先把数组用sort()排好序，会好做一些

def threeSum(self, nums):
    res = []
    nums.sort() # 数组排序
    for i in xrange(len(nums)-1):
        # 同一个数字只遍历一次，这样既能防止重复的数组出现，也能使同一个数组可以包含两个相同的数，这个方法必须建立在已经排好序的数组上
        # [-1,-1,0,1]，这个数组，用这个算法就能顺利解出答案
        # 注意[0,0,0]也是个坑
        if i>0 and nums[i] == nums[i-1]:    
            continue
        l, r = i+1, len(nums)-1 # 再剩下的数组中遍历
        while l<r:
            s = nums[i] + nums[l] + nums[r]
            # 根据结果的大小来判断怎么缩小区域
            if s < 0:
                l+=1
            elif s > 0:
                r-=1
            else:
                res.append([nums[i], nums[l], nums[j]])
                while l<r and nums[l]==nums[l+1]:   #防止重复数组
                    l+=1
                while l<r and nums[r]==nums[r-1]:
                    r-=1
                l+=1
                r-=1
    return res
