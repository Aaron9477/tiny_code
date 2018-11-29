input = [3,8,5,3,5,2,4]

def solution(input):
    if len(input) == 0:
        return None
    if len(input) == 1:
        return input[0]
    if len(input) == 2:
        return max(input[0], input[1])
    if len(input) == 3:
        return max(input[1], input[0]*input[2])

    res = []
    # Flase当前索引前一个没有值, True当前索引前一个有值
    res.append((0, False))
    res.append((input[0], True))
    res.append((input[0], False) if input[0]>input[1] else (input[1], True))

    for i in range(2, len(input)):
        if res[i-1][1] == False:
            res.append((res[i-1][0]+input[i], True))
        else:
            if res[i-2][0]+input[i] > res[i-1][0]:
                res.append((res[i-2][0]+input[i], True))
            else:
                res.append((res[i-1][0], False))
    return res[-1][0]

if __name__ == '__main__':
    print(solution(input))