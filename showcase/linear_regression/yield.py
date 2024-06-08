a = (x for x in range(10))
print(next(a))
print(next(a))
print(next(a))

a = (x for x in range(10))
for i in a:
    print(i)


def odd():
    yield 1
    yield 3
    yield 5


def odd2():
    a = 1
    while True:
        yield a
        a += 2


a = odd()
print('odd()', next(a))
print('odd()', next(a))
print('odd()', next(a))

a = odd2()
for i in range(10):
    print('odd2', next(a))


def fib():
    yield 1
    yield 1
    left = 1
    right = 1

    while True:
        ne = left + right
        yield ne
        left = right
        right = ne


a = fib()

for i in range(10):
    print(next(a))
