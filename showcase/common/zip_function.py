list1 = list(range(3))
print(list1)

for i in zip(list1):
    print(i)

list2 = list(range(3, 6))
print(list2)

for i in zip(list1, list2):
    print(i)
