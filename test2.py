class ListNode:
    def __init__(self, val):
        self.val = val
        self.next = None


a, b, c, d, e = ListNode(1), ListNode(2), ListNode(3), ListNode(4), ListNode(5)
a.next = b
b.next = c
c.next = d
d.next = e


from random import randint
class Solution:
    def __init__(self, head):
        self.head = head
    def get_random(self):
        node_count = 0
        p = self.head
        while p is not None:
            node_count += 1
            rand = randint(1, node_count)
            if rand == node_count:
                res = p.val
            p = p.next
        return res
            
s = Solution(a)
from collections import defaultdict
count=defaultdict(int)
for _ in range(1000000):
    res = s.get_random()
    count[res] += 1
print(count)