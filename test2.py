def core(n):
    def index(i,cur):
        if i == -1:
            return list(range(1, n + 1))
        else:
            res = []
            for i in nums:
                if i not in visited and abs(cur[-1] - i) != 1:
                    res.append(i)
            return res

    def backtrack(i, cur):
        if len(cur) == len(nums):
            res.append(cur.copy())
            return
        newindex = index(i,cur)
        for x in newindex:
            visited.add(x)
            cur.append(x)
            backtrack(x,cur)
            visited.remove(x)
            cur.pop()


    visited = set()
    nums = list(range(1, n + 1))
    res=[]
    backtrack(-1, [])
    for i in res:
        print(' '.join(list(map(str,i))))



if __name__ == '__main__':
    # n=int(input())
    core(10)