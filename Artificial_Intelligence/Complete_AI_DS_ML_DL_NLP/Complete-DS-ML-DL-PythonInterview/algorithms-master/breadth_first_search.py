from collections import deque

graph = {}
graph['you'] = ['alice', 'bob', 'claire']
graph['alice'] = ['peggy']
graph['bob'] = ['anuj']
graph['claire'] = ['jonny', 'thom']
graph['peggy'] = []
graph['anuj'] = []
graph['jonny'] = []
graph['thom'] = []


def is_person_seller(name):
    '''look for a person using arbitrary criteria'''
    return name[-1] == 'z'


def search(name, graph):
    '''Implementing breadth-first search algorithm'''
    search_queue = deque()
    search_queue += graph[name]
    searched_people = []

    while search_queue:
        person = search_queue.popleft()
        if person not in searched_people:
            if is_person_seller(person):
                return f'{person} is seller'
            else:
                searched_people.append(person)
        search_queue += graph[person]

    return 'There is no seller within your network'
