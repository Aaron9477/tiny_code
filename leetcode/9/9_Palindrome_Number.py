# coding=utf-8

class solution(object):
    """docstring for isPalindrome"""
    def isPalindrome(self, x):
        return `x`[::-1] == `x` #注意这是负号！！！！！！！！！！！！！！！！！

if __name__ == '__main__':
    a = solution()
    print(a.isPalindrome(12314321))