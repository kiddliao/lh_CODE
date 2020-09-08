def core(s):
    i = 0
    while i < len(s):
        if s[:i + 1] * (len(s) // len(s[:i + 1])) == s:
            return s[:i + 1]
        i += 1
def gcd(a, b):
    if b != 0:
        return gcd(b, a % b)
    else:
        return a
if __name__ == '__main__':
    s1 = 'abcabc'
    s2 = 'abcabcabcabc'
    t1, t2 = core(s1), core(s2)
    n1, n2 = len(s1) // len(t1), len(s2) // len(t2)
    if t1 != t2:
        print('')
    else:
        n = gcd(n1, n2)
        print(t1 * n)
        
