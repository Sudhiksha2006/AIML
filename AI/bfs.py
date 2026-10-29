import pandas as pd
from collections import deque

data = pd.read_csv("graph.csv")
graph = {}

for _, r in data.iterrows():
    graph.setdefault(r.Source, []).append(r.Destination)
    graph.setdefault(r.Destination, [])

def bfs(start, goal):
    q = deque([[start]])
    visited = {start}

    while q:
        path = q.popleft()
        node = path[-1]

        if node == goal:
            return path

        for n in graph[node]:
            if n not in visited:
                visited.add(n)
                q.append(path + [n])

    return None

print("BFS Path:", bfs("A", "F"))