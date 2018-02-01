# coding=utf-8

# class Solution(object):
#     """docstring for Solution"""
#     def combinationSum3(self, k, n):
#         self.result = []
#         if n == 1:
#             self.result = [[k]]
#         elif n == 2:
#             for i in range(1,int((k+1)/2)):
#                 for j in range(i+1,k+1):
#                     if i+j == n:
#                         self.result.append([i,j])
#                         break
#         for i in range(1,3):
#             for j in range(i+1, n+1):
#                 l = n-i-j
#                 if l > j:
#                     self.result.append([i,j,l])
#                     continue
#                 else:
#                     break
#         return self.result
# 错误！！！！！！！！！！！！！！！！！！！！！！！！！！！！没做出来       
class Solution(object):
    """docstring for Solution"""
    def combinationSum3(self, k, n):
        self.result = []
        if n > 55:
            return self.result
        elif n == 2:
            for i in range(1,int((k+1)/2)):
                for j in range(i+1,k+1):
                    if i+j == n:
                        self.result.append([i,j])
                        break
        for i in range(1,3):
            for j in range(i+1, n+1):
                l = n-i-j
                if l > j:
                    self.result.append([i,j,l])
                    continue
                else:
                    break
        return self.result


if __name__ == '__main__':
    print reduce(lambda x,y:x+y, [i for i in range(1,101)])
    a = Solution()
    k = 2
    n = 9
    print(a.combinationSum3(k,n))
