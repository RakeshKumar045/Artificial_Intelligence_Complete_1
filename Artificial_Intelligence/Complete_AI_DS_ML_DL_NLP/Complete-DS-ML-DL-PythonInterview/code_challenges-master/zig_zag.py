'''
A sequence of integers is called a zigzag sequence if each of its elements is
either strictly less than all its neighbors or strictly greater than all its
neighbors. For example, the sequence 4 2 3 1 5 3 is a zigzag, but 7 3 5 5 2
and 3 8 6 4 5 aren't. Sequence of length 1 is also a zigzag.

For a given array of integers return the length of its longest contiguous
sub-array that is a zigzag sequence.

Example

For a = [9, 8, 8, 5, 3, 5, 3, 2, 8, 6], the output should be
zigzag(a) = 4.

The longest zigzag sub-arrays are [5, 3, 5, 3] and [3, 2, 8, 6] and they both
have length 4.

For a = [4, 4], the output should be
zigzag(a) = 1.

The longest zigzag sub-array is [4] - it has only one element, which is strictly
greater than all its neighbors (there are none of them).

Input/Output

[execution time limit] 4 seconds (py3)

[input] array.integer a

Guaranteed constraints:
2 ≤ a.length ≤ 25,
0 ≤ a[i] ≤ 100.
'''


def zigzag(a):
    best_ans = []
    longest_ans = 0
    temp = [a[0]]

    for i in range(1, len(a)):
        if len(temp) == 1:
            if a[i] != temp[-1]:
                temp.append(a[i])
            else:
                temp = [a[i]]
        else:
            if (temp[-1] < temp[-2] and temp[-1] < a[i]) or (temp[-1] > temp[-2] and temp[-1] > a[i]):
                temp.append(a[i])
            else:
                if a[i] != temp[-1]:
                    temp = [temp[-1]] + [a[i]]
                else:
                    temp = [a[i]]
        if len(temp) > longest_ans:
            longest_ans = len(temp)
            best_ans = temp

    return longest_ans
