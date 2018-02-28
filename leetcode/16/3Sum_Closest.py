class Solution(object):
    def threeSumClosest(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: int
        """
        # res = []
        if len(nums) < 3:
            return
        nums.sort()
        result = nums[0] + nums[1] + nums[2]
        diff_tar = abs(result - target)
        # 固定第一个数字，另外两个数字分别从第一个数的右端和最后端向中间靠拢
        for i in range(len(nums) - 1):  # 这里要减一，因为后面要取两个数字
            if i > 0 and nums[i] == nums[i - 1]:    # 略过相同的数字
                i += 1
            l, r = i + 1, len(nums) - 1
            while l < r:
                diff_now = nums[i] + nums[l] + nums[r] - target
                diff_abs = abs(diff_now)
                if diff_abs < diff_tar:
                    diff_tar = diff_abs
                    result = nums[i] + nums[l] + nums[r]
                    # res.clear()
                    # res.append([nums[i],nums[l],nums[r]])
                # elif diff_abs == diff_tar:
                #     res.append([nums[i],nums[l],nums[r]])
                # 因为要变动两个数字，所以先固定一个，变动其中一个
                # 通过判断大小，可以减小计算量，不需要求和比较的略过
                # 如[-3,-2,-1,0,1] -3-2+1<0 就不会再对[-3,-2,0]求和
                if diff_now <= 0:
                    l += 1
                    # 移动之后如果数字相同接着移动
                    while l < r and nums[l] == nums[l - 1]:  # 这里必须加上l<r，否则数组会溢出
                        l += 1
                elif diff_now > 0:
                    r -= 1
                    while l < r and nums[r] == nums[r + 1]:
                        r -= 1
        return result


