'''Given a matrix (i.e. an array of arrays), find its submatrix obtained by
deleting the specified rows and columns.
Example
For matrix = [[1, 0, 0, 2], 
              [0, 5, 0, 1], 
              [0, 0, 3, 5]]
rowsToDelete = [1], and columnsToDelete = [0, 2], the output should be

constructSubmatrix(matrix, rowsToDelete, columnsToDelete) = [[0, 2],
                                                             [0, 5]]

[input] array.array.integer matrix

2-dimensional array of integers.

Guaranteed constraints:
1 ≤ matrix.length ≤ 5,
1 ≤ matrix[0].length ≤ 5,
0 ≤ matrix[i][j] ≤ 15.

[input] array.integer rowsToDelete

Array of indices (0-based) of rows to be deleted.

Guaranteed constraints:
0 ≤ rowsToDelete.length ≤ 2,
0 ≤ rowsToDelete[i] < matrix.length.

[input] array.integer columnsToDelete

Array of indices (0-based) of columns to be deleted.

Guaranteed constraints:
0 ≤ columnsToDelete.length ≤ 2,
0 ≤ columnsToDelete[i] < matrix[0].length.
'''


def construct_submatrix(matrix, rowsToDelete, columnsToDelete):
    i = 0
    for row in rowsToDelete:
        if i == 0:
            matrix.pop(row)
        else:
            matrix.pop(row - 1)
        i += 1

    for l in range(len(matrix)):
        i = 0
        for col in columnsToDelete:
            if i == 0:
                matrix[l].pop(col)
            else:
                matrix[l].pop(col - 1)
            i += 1

    return matrix
