class Solution:
    def __init__(self):
        self.dict = {}
    def save(self,s):
        s = s.strip().split(',')
        for k_v in s:
            k, v = k_v.split(':')
            v = int(v)
            self.dict[v] = k
    def find_lucky_num(self,x):
        return self.dict[x]

if __name__ == '__main__':
    a = Solution()
    s = input()
    lucky_num = int(input())
    a.save(s)
    res=a.find_lucky_num(lucky_num)
    print(res)
