'''
Implementing some funtions to compute measurements recursively.
'''


def find_gcd(a, b):
    '''Find greatest common divisor of `a` and `b`.'''
    if a == 0:
        return b
    elif b == 0:
        return a
    else:
        if a < b:
            a, b = b, a
        remainder = a % b
        return find_gcd(b, remainder)


def fact(n):
    '''Compute n factorial.'''
    if n < 0 or type(n) == 'float':
        raise ValueError('n should be positive integer.')
    if n == 0 or n == 1:
        return 1
    else:
        return n * fact(n - 1)


def compute_sum(L):
    '''Compute the sum of element in `L` recursively.'''
    if len(L) == 0:
        return 0
    elif len(L) == 1:
        return L[0]
    else:
        return L[0] + compute_sum(L[1:])


def count_elements(L):
    '''Count number of element in `L` recursively.'''
    if len(L) == 0:
        return 0
    elif len(L) == 1:
        return 1
    else:
        return 1 + count_elements(L[1:])


def find_max(L):
    if len(L) == 1:
        return L[0]
    elif len(L) == 2:
        return L[0] if L[0] > L[1] else L[1]
    return L[0] if L[0] > find_max(L[1:]) else find_max(L[1:])
