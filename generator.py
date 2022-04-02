from typing import List, Tuple, MutableMapping, Optional

import numpy as np

class Workbench:

    def __init__(self, ntype: int) -> None:
        self.table : MutableMapping[Tuple[int, int], int] = {}
        self.expandable_queue: List[Tuple[int, int]] = []
        self.neighbors = [(0,1), (1,0), (0,-1), (-1,0), (1,1), (-1,-1), (1,-1), (-1,1)]
        self.ntype = ntype
        self.random(0,0)

    def get(self, x: int, y: int) -> Optional[int]:
        if (x, y) not in self.table:
            return None
        return self.table[(x, y)]

    def set(self, x: int, y: int, pin: int) -> None:
        assert self.get(x, y) is None
        self.table[(x, y)] = pin
        if self.is_expandable(x, y):
            self.expandable_queue.append((x, y))
        for nx, ny in self.neighbors:
            coords = (x + nx, y + ny)
            if coords in self.expandable_queue and not self.is_expandable(*coords):
                self.expandable_queue.remove(coords)

    def random(self, x: int, y: int) -> None:
        self.set(x, y, np.random.randint(0, self.ntype))

    def get_expansions(self, x: int, y: int) -> List[Tuple[int, int]]:
        exps = []
        for nx, ny in self.neighbors:
            coords = (x + nx, y + ny)
            if self.get(*coords) is None:
                exps.append(coords)
        return exps

    def is_expandable(self, x: int, y: int) -> bool:
        return len(self.get_expansions(x, y)) > 0

    def next(self) -> Tuple[int, int]:
        x,y = self.expandable_queue[int(np.random.randint(0, len(self.expandable_queue)))]
        exps = self.get_expansions(x, y)
        assert len(exps) > 0
        coords = exps[int(np.random.randint(0, len(exps)))]
        assert self.get(*coords) is None
        return coords

    def is_neighbor(self, coords1: Tuple[int, int], coords2: Tuple[int, int]) -> bool:
        return abs(coords1[0] - coords2[0]) <= 1 and abs(coords1[1] - coords2[1]) <= 1

    def numpy(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        items = np.zeros([len(self.table), self.ntype])
        for idx, pin in enumerate(self.table.values()):
            items[idx, pin] = 1.
        nodes = np.zeros([len(self.table), self.ntype])
        for idx, pin in enumerate(self.table.values()):
            nodes[idx, pin] = 1.
        edges = np.zeros([len(self.table), len(self.table), 1])
        for i, coords1 in enumerate(self.table.keys()):
            for j, coords2 in enumerate(self.table.keys()):
                if i == j:
                    continue
                if self.is_neighbor(coords1, coords2):
                    edges[i, j, 0] = 1.
                    edges[j, i, 0] = 1.
        coords2D =  np.array([[x,y] for x,y in self.table.keys()])
        return items, nodes, edges, coords2D


def generate(ntypes: int, npins: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    workbench = Workbench(ntypes)
    for _ in range(npins - 1):
        coords = workbench.next()
        workbench.random(*coords)
    return workbench.numpy()


def print_graph(nodes: np.ndarray, coords2D: np.ndarray) -> None:
    minx = coords2D[:,0].min()
    miny = coords2D[:,1].min()
    maxx = coords2D[:,0].max()
    maxy = coords2D[:,1].max()
    rangex = maxx - minx + 1
    rangey = maxy - miny + 1
    array = np.full((rangex, rangey), -1)
    for idx, coord in enumerate(coords2D):
        array[coord[0] - minx, coord[1] - miny] = nodes[idx].argmax()
    txt = ""
    for i in range(array.shape[0]):
        for j in range(array.shape[1]):
            if array[i,j] < 0:
                txt += "."
            else:
                txt += str(array[i,j])
        txt += "\n"
    print(txt)


if __name__ == "__main__":
    items, nodes, edges, coords2D = generate(6, 1000)
    print_graph(nodes, coords2D)
