import pandas as pd

data = pd.read_csv("edges.csv")
hdata = pd.read_csv("heuristic.csv")

h = dict(zip(hdata.Node, hdata.H))
graph = {}

for _, r in data.iterrows():
    graph.setdefault(r.Source, []).append((r.Destination, r.Cost))
    graph.setdefault(r.Destination, [])

def astar(start, goal):
    paths = [(h[start], 0, [start])]

    while paths:
        paths.sort()
        f, cost, path = paths.pop(0)
        node = path[-1]

        if node == goal:
            return path

        for nxt, c in graph[node]:
            if nxt not in path:
                g = cost + c
                paths.append((g + h[nxt], g, path + [nxt]))

    return None

print("Path:", astar("A", "D"))