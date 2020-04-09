'''
Digit root of some positive integer is defined as the sum of all of its digits.

You are given an array of integers. Sort it in such a way that if a comes before
b then the digit root of a is less than or equal to the digit root of b. If two
numbers have the same digit root, the smaller one (in the regular sense) should
come first. For example 4 and 13 have the same digit root, however 4 < 13 thus
4 comes before 13 in any digitRoot sorting where both are present.

Example

For a = [13, 20, 7, 4], the output should be [20, 4, 13, 7].

[execution time limit] 4 seconds (py3)

[input] array.integer a

Array of positive integers.
'''


def digit_root_sort(a):
    L = [str(n) for n in a]
    ans_dict = {}
    for n in L:
        temp = sum([int(d) for d in n])
        ans_dict[temp] = ans_dict.get(temp, []) + [int(n)]
    ans = []
    for digits in sorted(ans_dict.keys()):
        temp = sorted(ans_dict[digits])
        for n in temp:
            ans.append(n)
    return ans
