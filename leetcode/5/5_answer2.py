class Solution:
    # @return a string
    def longestPalindrome(self, s):
        # 回文字符串的增长，一定是基于其子回文字符串 如abcba是基于子回文bcb
        # 所以只需要按顺序把字符串全扫描一遍，就能得到答案
        if len(s) == 0:
            return 0
        maxlen = 1  # 第一次遇到长的回文，maxlen会慢慢变大，后来必须遇到更长的回文才会更新
        start = 0
        for i in range(len(s)):
            # 针对abcba，扫描到某一个回文字符串的尾部并且长度比maxlen大，才会被记录，先扫描到第四个字符b，然后得到bcb，再扫描下一个得到abcba
            # 一次加两个字符的回文字符 bb->abba 或者 aba->cabac
            if i-maxlen>=1 and s[i-maxlen-1:i+1] == s[i-maxlen-1:i+1][::-1]:
                start = i-maxlen-1  # 记录字符串开头 即上一行第二个判断开头部分 start是用来找到完整的字符串并且返回的
                maxlen += 2 # 因为i往前移动1，这是是回文字符串，长度要加2
                continue
            # 一次加一个字符的回文字符 a->aa
            if i-maxlen>=0 and s[i-maxlen:i+1] == s[i-maxlen:i+1][::-1]:
                start = i-maxlen    # 记录字符串开头 即上一行第二个判断开头部分
                maxlen += 1
        return s[start:start+maxlen]

        # 收获： 遇到问题先考虑从头到尾扫描一遍字符串能否达到目的，不要上来就想复杂的方法