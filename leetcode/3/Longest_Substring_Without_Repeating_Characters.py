class Solution(object):
    def lengthOfLongestSubstring(self, s):
        """
        :type s: str
        :rtype: int
        """
        if len(s) <= 1:
            return len(s)
        tmp_length = 0
        length = 0
        char_dict = []
        # i = 0
        for j in range(len(s)):
            i = j
            while i < len(s) and not s[i] in char_dict:
                char_dict.append(s[i])
                i += 1
                tmp_length += 1
            if tmp_length > length:
                length = tmp_length
            char_dict = []
            tmp_length = 0
        return length



# 程序运行时间过长，只有最后一个案例没过