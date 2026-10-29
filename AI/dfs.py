import pandas as pd

data = pd.read_csv("graph.csv")
graph = {}

for _, r in data.iterrows():
    graph.setdefault(r.Source, []).append(r.Destination)
    graph.setdefault(r.Destination, [])

def dfs(start, goal):
    stack = [[start]]
    visited = set()

    while stack:
        path = stack.pop()
        node = path[-1]

        if node == goal:
            return path

        if node not in visited:
            visited.add(node)
            for n in reversed(graph[node]):
                if n not in visited:
                    stack.append(path + [n])

    return None

print("DFS Path:", dfs("A", "F"))