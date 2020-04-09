'''
Implementing binary and simple search algorithms.
'''


def binary_search(L, element):
    '''
    Check if `element` is in `L` and return its position using binary search
    algorithm; otherwise, return None if not found. Assumes `L` is a sorted
    list.
    '''
    low = 0
    high = len(L) - 1
    while high >= low:
        mid = (low + high) // 2
        guess = L[mid]
        if guess > element:
            high = mid - 1
        elif guess < element:
            low = mid + 1
        else:
            return mid
    return None


def simple_search(L, element):
    '''
    Check if `element` is in `L` and return its position using simple search
    algorithm; otherwise, return None if not found.
    '''
    for i, e in enumerate(L):
        if e == element:
            return i
    return None
