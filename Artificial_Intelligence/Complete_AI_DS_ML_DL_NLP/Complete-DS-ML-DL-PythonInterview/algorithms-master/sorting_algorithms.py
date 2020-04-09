'''
Implement sorting algorithms.
'''


def selection_sort(L):
    '''
    Sort `L` array in ascending order.
    '''
    sorted_array = []
    for _ in range(len(L)):
        # Use element at position 0 as a starting point
        smallest_element = L[0]
        smallest_index = 0
        for i in range(1, len(L)):
            if smallest_element > L[i]:
                smallest_element = L[i]
                smallest_index = i
        # Add element to sorted_array and delete it from the input array L
        sorted_array.append(L.pop(smallest_index))
    return sorted_array


def quick_sort(L):
    '''
    Sort `L` array in ascending order using first element of an array as the
    pivot.'''
    # empty array or array with length zero is a sorted array
    if len(L) < 2:
        return L
    else:
        # Choose first element as pivot that will divide L into two sub-arrays
        pivot = L[0]
        left = [e for e in L[1:] if e <= pivot]
        right = [e for e in L[1:] if e > pivot]
        return quick_sort(left) + [pivot] + quick_sort(right)


def quick_sort_2(L):
    '''
    Sort `L` array in ascending order using random element from the array as
    the pivot.'''
    L = list(L)
    import random
    # empty array or array with length zero is a sorted array
    if len(L) < 2:
        return L
    else:
        # Choose first element as pivot that will divide L into two sub-arrays
        pivot_index = random.randint(0, len(L))
        pivot = L[pivot_index]
        L.remove(pivot)
        left = [e for e in L if e <= pivot]
        right = [e for e in L if e > pivot]
        return quick_sort(left) + [pivot] + quick_sort(right)
