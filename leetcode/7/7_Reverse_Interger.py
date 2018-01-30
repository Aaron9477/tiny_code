# coding=utf-8

class Sulotion(object):
    """docstring for Reverse"""
    def reverse(self, arg):
        self.limit = 2**31  # 只能在函数内定义，或者在__init__中定义
        if self.judge(arg):
            return 0
        if arg<0:
            arg = -arg
            arg = -self.handle(arg)
        else:
            arg = self.handle(arg)
        return arg

    def handle(self, input):
        input_str = str(input)  # 转化成string来做
        output = '' # 初始化
        for i in range(len(input_str)):
            output += input_str[len(input_str)-1-i] # 注意要减1
        output = int(output)
        if self.judge(output):  # 输出也要判断！！！！！！！！！！！！！
            output = 0
        return output

    def judge(self, input):
        if input>self.limit or input<-self.limit:
            return True

if __name__ == '__main__':
    a = Sulotion()
    result = a.reverse(1534234356)
    print(result)