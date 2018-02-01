# coding=utf-8
# 纪念一下第一次提交一次就通过！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！
class Solution(object):
    """docstring for Solution"""
    def combinationSum(self, candidates, target):
        # handled = self.handle(candidates, target)
        handled = [x for x in candidates if x<=target]  # 此处的写法要注意
        res = []
        res = self.helpSum(handled, target, [], res, 0)
        return res


    # def handle(self, candidates, target):
    #     for i in xrange(len(candidates)):   # 注意，如果用del删除数组元素，删除一个之后后面的索引也变了，所以del不是很安全
    #         if candidates[i] > target:
    #             candidates[i]
    #     return candidates

    def helpSum(self, input, target, arr, res, curr):
        if sum(arr) > target:
            return
        if sum(arr) == target:
            res.append(list(arr))
            return
        for i in xrange(curr, len(input)):  # 这里就是input的长度，因为需要的是索引值，不需要减一
            arr.append(input[i])
            self.helpSum(input, target, arr, res, i)
            arr.pop()
        return res

if __name__ == '__main__':
    a = Solution()
    candidates = [2,3,4,7,8]
    target = 7
    print(a.combinationSum(candidates, target))