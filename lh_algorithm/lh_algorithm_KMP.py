def make_next(s):
    n = len(s)
    next = [-1] * (n+1)
    next[1] = 0
    i, j = 1, 0
    while i < n:
        if j == -1 or s[i] == s[j]:
            i += 1
            j += 1
            next[i] = j
        else:
            j = next[j]
    return next[:-1]

def KMP_match(s, p):
    # 传入一个母串和一个子串
    # 返回子串匹配上的第一个位置，若没有匹配上返回-1
    if not s: return - 1
    if not p: return 0
    next = make_next(p)
    sp = pp = 0
    while sp < len(s) and pp < len(p):
        if pp == -1 or s[sp] == p[pp]:
            sp, pp = sp + 1, pp + 1
        else:
            pp = next[pp]
    if pp == len(p):  #匹配成功
        return sp-pp

def KMP_matchall(s, p):
    cur = []
    s = list(s)
    while 1:
        next = make_next(p)
        sp = pp = 0
        while sp < len(s) and pp < len(p):
            if pp == -1 or s[sp] == p[pp]:
                sp, pp = sp + 1, pp + 1
            else:
                pp = next[pp]
        if pp == len(p):  #匹配成功
            cur.append(sp - pp)
            s[sp - pp] = '*'
        else: break
    return cur

print(KMP_match('faoboofarboo','faoboo'))      
print(KMP_matchall('barfoobarthefoobarman','foobar'))      