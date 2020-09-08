class B(object):
    score = 0
    def __init__(self, name, top):
        self.name = name
        self.__top = top

    #将方法当作属性
    @property
    def get_top(self):
        return self.__top

    @get_top.setter
    def get_top(self, val):
        self.__top = val

    @staticmethod
    def prime():
        print('666')

    @classmethod
    def prime2(cls):
        print(cls.score)
        print(cls('李四', 2).get_top)


b = B('张三', 10)
print(b.get_top)
b.get_top = 9
print(b.get_top)
print(hasattr(b, 'get_top'))  # 对象有没有同名的属性或者方法

B('李四', 2).prime()
B('李四', 2).prime2()
b.prime2()
