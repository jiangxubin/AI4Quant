from collections import Iterable


class MyIterator:
    def __init__(self, end):
        self.i = 0
        self.end = end

    def __iter__(self):
        return self

    def __next__(self):
        if self.i <= self.end:
            out = self.i
            self.i += 1
            return out
        else:
            raise StopIteration


a = list(MyIterator(5))


def fei_generator():
    a = 0
    b = 1
    while True:
        a, b = b, a+b
        yield a


def fei_generator2():
    a = 0
    b = 1
    yield a
    a, b = b, a+b
    yield a


fei = fei_generator2()
for item in fei:
    if item>20:
        break
    print(item)

