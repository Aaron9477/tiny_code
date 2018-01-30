#coding=utf-8

class solution(object):
    """docstring for solution"""
    def reverse(self, x):
        s = cmp(x,0)
        r = int(`s*x`[::-1])    # ``转化成string型，[::1]将列表或者字符倒过来
        return s*r*(r<2**31)

if __name__ == '__main__':
    a = solution()
    print(a.reverse(2310))
