# 开始是打算把数组取反，挨个比较，发现不合适，因为反向之后回文的子字符串所在位置并不相同
# 这个程序能做到复杂度为O(n2)
class Solution(object):
    def longestPalindrome(self, s):
        """
        :type s: str
        :rtype: str
        """
        res = ''
        for i in xrange(len(s)):
            temp = self.helper(s, i, i) # 回文的第一种形式
            if len(temp) > len(res):
                res = temp[:]
            temp = self.helper(s, i, i + 1) # 回文的第二种形式
            if len(temp) > len(res):
                res = temp[:]
        return res

    def helper(self, s, l, r):
        while l >= 0 and r < len(s) and s[l] == s[r]:
            l -= 1
            r += 1
        return s[l + 1:r]  # 此处l+1