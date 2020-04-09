'''
Given an integer size, return an array containing each integer from 1 to size in the following order:

1, size, 2, size - 1, 3, size - 2, 4, ...

Example

For size = 7, the output should be
constructArray(size) = [1, 7, 2, 6, 3, 5, 4].

Input/Output

[execution time limit] 4 seconds (py3)

[input] integer size

A positive integer.

Guaranteed constraints:
1 ≤ size ≤ 15.
'''


def construct_array(size):
    if size == 1:
        return [size]
    array = list(range(1, size))
    reverse_array = list(range(size, 1, -1))
    ans = []

    for i in range(len(reverse_array)):
        element = array[i]
        reverse_element = reverse_array[i]
        if element in ans:
            break
        ans.append(element)
        if reverse_element in ans:
            break
        else:
            ans.append(reverse_element)

    return ans
