class Solution(object):
    def lengthOfLongestSubstring(self, s):
        """
        :type s: str
        :rtype: int
        """
        start = maxlength = 0
        used = {}
        for i, iterm in enumerate(s):
            # 这里要判断start和当前索引的关系，有可能之前某一个字符的重复，减少了一部分字符，如abcabe，因为a的重复，前半段abc都被删去
            if iterm in used and start <= used[iterm]:
                start = used[iterm] + 1
            else:
                maxlength = max(maxlength, i-start+1)
            used[iterm] = i
        return maxlength