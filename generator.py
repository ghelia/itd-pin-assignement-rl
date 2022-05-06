from typing import List, Tuple, MutableMapping, Optional

import torch
import numpy as np

from config import Config

class Workbench:

    def __init__(self, ntype: int) -> None:
        self.table : MutableMapping[Tuple[int, int], int] = {}
        self.expandable_queue: List[Tuple[int, int]] = []
        # self.neighbors = [(0,1), (1,0), (0,-1), (-1,0), (1,1), (-1,-1), (1,-1), (-1,1)]
        self.neighbors = [(0,1), (1,0), (0,-1), (-1,0)]
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
        diff = (coords1[0] - coords2[0], coords1[1] - coords2[1])
        return diff in self.neighbors

    def numpy(self, overlap_ratio: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        items_type = np.zeros([len(self.table), self.ntype])
        items_neighbors = np.zeros([len(self.table), self.ntype])
        for idx, pin in enumerate(self.table.values()):
            items_type[idx, pin] = 1.
        for idx, (x, y) in enumerate(self.table.keys()):
            for nx, ny in self.neighbors:
                coords = (x + nx, y + ny)
                ntype = self.get(*coords)
                if ntype is not None:
                    items_neighbors[idx, ntype] = 1.
        items = np.concatenate([items_type, items_neighbors], axis=1)

        nodes = np.zeros([len(self.table), self.ntype])
        for idx, pin in enumerate(self.table.values()):
            nodes[idx, pin] = 1.
        for idx in range(nodes.shape[0]):
            if np.random.random() <= overlap_ratio:
                rtype  = np.random.randint(nodes.shape[1])
                nodes[idx][rtype] = 1.
        node_placements = np.zeros([len(self.table), self.ntype*2 + 1])
        nodes = np.concatenate([nodes, node_placements], axis=1)

        edges = np.zeros([len(self.table), len(self.table)])
        for i, coords1 in enumerate(self.table.keys()):
            for j, coords2 in enumerate(self.table.keys()):
                if i == j:
                    continue
                if self.is_neighbor(coords1, coords2):
                    edges[i, j] = 1.
                    edges[j, i] = 1.
        coords2D = np.array([[x,y] for x,y in self.table.keys()])
        np.random.shuffle(items)
        return items, nodes, edges, coords2D


def generate(ntypes: int, npins: int, overlap_ratio: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    workbench = Workbench(ntypes)
    for _ in range(npins - 1):
        coords = workbench.next()
        workbench.random(*coords)
    return workbench.numpy(overlap_ratio)


def get_node_type(node: np.ndarray) -> int:
    node_type_ohv = node[Config.placement_offset:Config.placement_offset + Config.ntypes]
    tsum = node_type_ohv.sum()
    if tsum == 0:
        return -1
    assert tsum == 1
    return int(node_type_ohv.argmax())

def have_item(node: np.ndarray) -> bool:
    return node[Config.placed_flag_index] == 1.

def check_neighbor_compatible(node: np.ndarray, neighbor: np.ndarray) -> bool:
    if not have_item(node):
        return True
    if not have_item(neighbor):
        return True
    neighbor_type = get_node_type(neighbor)
    assert neighbor_type >= 0
    return node[Config.possible_neighbor_offset + neighbor_type] == 1.


def check_placement(place: int,
                    nodes: np.ndarray,
                    edges: np.ndarray,
                    check_neighbors_type: bool = True
                   ) -> bool:
    node = nodes[place]
    node_possible_types = node[:Config.ntypes]
    placed_item_type = get_node_type(node)
    assert placed_item_type >= 0
    if node_possible_types[placed_item_type] == 0:
        return False
    if not check_neighbors_type:
        return True
    edge = edges[place]
    for idx in range(Config.nitems):
        if edge[idx] != 0:
            neighbor = nodes[idx]
            if not check_neighbor_compatible(node, neighbor):
                return False
            if not check_neighbor_compatible(neighbor, node):
                return False
    return True


def put_items(selections: torch.Tensor,
              places: torch.Tensor,
              items: torch.Tensor,
              nodes: torch.Tensor
             ) -> None:
    batch = len(selections)
    nodes[np.arange(batch),places, Config.placed_flag_index] = 1.
    nodes[np.arange(batch),places, Config.placement_offset:] = items[np.arange(batch),selections]


def check_placements(places: torch.Tensor,
                     nodes: torch.Tensor,
                     edges: torch.Tensor,
                     check_neighbors_type: bool = True
                   ) -> List[bool]:
    results: List[bool] = []
    assert len(places) == len(nodes) == len(edges)
    for idx in range(len(places)):
        results.append(check_placement(
            int(places[idx].int().item()),
            nodes[idx].cpu().numpy(),
            edges[idx].cpu().numpy(),
            check_neighbors_type
        ))
    return results


def batch(batch_size: int = Config.batch_size,
          ntypes: int = Config.ntypes,
          npins: int = Config.nitems,
          overlap_ratio: float = Config.overlap_ratio,
          return_coords: bool = False
         ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    instances = [generate(ntypes, npins, overlap_ratio) for _ in range(batch_size)]
    items = np.stack([instance[0] for instance in instances])
    nodes = np.stack([instance[1] for instance in instances])
    edges = np.stack([instance[2] for instance in instances])
    if not return_coords:
        return (
            torch.tensor(items, device=Config.device, dtype=torch.float32),
            torch.tensor(nodes, device=Config.device, dtype=torch.float32),
            torch.tensor(edges, device=Config.device, dtype=torch.bool)
        )
    coords = [instance[3]  for instance in instances]
    return (
        torch.tensor(items, device=Config.device, dtype=torch.float32),
        torch.tensor(nodes, device=Config.device, dtype=torch.float32),
        torch.tensor(edges, device=Config.device, dtype=torch.bool),
        coords
    )


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
    items, nodes, edges, coords2D = generate(6, 100, 0.25)
    print_graph(nodes, coords2D)
