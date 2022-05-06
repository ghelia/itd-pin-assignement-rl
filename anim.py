import torch
import numpy as np
import pygame
import time

from generator import *
from config import Config
from network import Agent

Config.batch_size = 10
Config.overlap_ratio = 0.33
Config.nitems = 80

colors = [
    np.array((0,200,0)),
    np.array((30,30,255)),
    np.array((255,0,255)),
    np.array((0,255,255)),
    np.array((255,255,0))
]


def draw_item(item, background, pos):
    radius = 20
    width = 4
    bottomx = -20
    bottomy = 25
    neighbor_gap = 12
    neighbor_radius = 5
    neighbor_width = 2
    itype = item[:Config.ntypes].argmax()
    pygame.draw.circle(background, colors[itype], pos, radius)
    pygame.draw.circle(background, (0,0,0), pos, radius, width=3)
    neighbor_idx = 0
    for neighbor_type, neighbor in enumerate(item[Config.ntypes:]):
        if neighbor == 1.:
            npos = (
                pos[0] + bottomx + neighbor_idx*neighbor_gap,
                pos[1] + bottomy
            )
            pygame.draw.circle(background, colors[neighbor_type], npos, neighbor_radius)
            pygame.draw.circle(background, (0,0,0), npos, neighbor_radius, width=neighbor_width)
            neighbor_idx += 1

def draw_items(items, background, actions):
    line_length = 7
    startx = 30
    starty = 30
    x = 0
    y = 0
    wgap = 55
    hgap = 70
    selected = [action[0] for action in actions]
    for idx, item in enumerate(items):
        pos = (startx + x * wgap, starty + y * hgap)
        if idx not in selected:
            draw_item(item, background, pos)
        x += 1
        if x >= line_length:
            x = 0
            y += 1


def draw_node(node, coord, background, pos, size):
    border = 3
    half = int(size/2)
    if node[:Config.ntypes].sum() == 1:
        nodetype = node[:Config.ntypes].argmax()
        pygame.draw.polygon(
            background,
            colors[nodetype] * 0.7,
            (
                (pos[0] - half, pos[1] - half),
                (pos[0] - half, pos[1] + half),
                (pos[0] + half, pos[1] + half),
                (pos[0] + half, pos[1] - half)
            )
        )
    else:
        assert node[:Config.ntypes].sum() == 2
        first = True
        for nodetype, typevalue in enumerate(node[:Config.ntypes]):
            if typevalue == 1:
                if first:
                    pygame.draw.polygon(
                        background,
                        colors[nodetype] * 0.7,
                        (
                            (pos[0] - half, pos[1] + half),
                            (pos[0] + half, pos[1] + half),
                            (pos[0] + half, pos[1] - half)
                        )
                    )
                    first = False
                else:
                    pygame.draw.polygon(
                        background,
                        colors[nodetype] * 0.7,
                        (
                            (pos[0] - half, pos[1] - half),
                            (pos[0] - half, pos[1] + half),
                            (pos[0] + half, pos[1] - half)
                        )
                    )
    pygame.draw.polygon(
        background,
        (0,0,0),
        (
            (pos[0] - half, pos[1] - half),
            (pos[0] - half, pos[1] + half),
            (pos[0] + half, pos[1] + half),
            (pos[0] + half, pos[1] - half)
        ),
        width=border,
    )

def draw_error(pos, background, size):
    half = int(size/2)
    border = 6
    pygame.draw.polygon(
        background,
        (255,0,0),
        (
            (pos[0] - half, pos[1] - half),
            (pos[0] - half, pos[1] + half),
            (pos[0] + half, pos[1] + half),
            (pos[0] + half, pos[1] - half)
        ),
        width=border,
    )
    pygame.draw.polygon(
        background,
        (255,0,0),
        (
            (pos[0] - half, pos[1] - half),
            (pos[0] - half, pos[1] + half),
            (pos[0] + half, pos[1] - half)
        ),
        width=border,
    )
    pygame.draw.polygon(
        background,
        (255,0,0),
        (
            (pos[0] - half, pos[1] - half),
            (pos[0] - half, pos[1] + half),
            (pos[0] + half, pos[1] + half)
        ),
        width=border,
    )

def draw_graph(items, nodes, coords, background, actions, step_rewards):
    center = (1200, 600)
    size = 80
    table = {int(action[1]):int(action[0]) for action in actions}
    steps = {idx:int(action[1]) for idx,action in enumerate(actions)}
    for idx in range(len(nodes)):
        node = nodes[idx]
        coord = coords[idx]
        reward = 0
        if idx < len(step_rewards):
            reward = step_rewards[idx]
        pos = (
            center[0] + size * coord[0],
            center[1] + size * coord[1]
        )
        draw_node(node, coord, background, pos, size)
        if idx in table.keys():
            draw_item(items[table[idx]], background, pos)

    for step,reward in enumerate(step_rewards):
        if reward < 0:
            idx = steps[step]
            coord = coords[idx]
            pos = (
                center[0] + size*coord[0],
                center[1] + size*coord[1]
            )
            draw_error(pos, background, size)


agent = Agent()
agent.train()
agent.load_state_dict(torch.load("./agent-5261-increase-size.chkpt"))
items, nodes, edges, coords = batch(npins=Config.nitems, batch_size=Config.batch_size, return_coords=True)
with torch.no_grad():
    _, _, _, _, actions, rewards, step_rewards = agent(items, nodes, edges, greedy=True, return_step_rewards=True)

items = items[0]
nodes = nodes[0]
edges = edges[0]
coords = coords[0]
actions = actions[0]
actions = [(actions[idx*2], actions[idx*2 + 1]) for idx in range(len(actions) // 2)]
rewards = rewards[0]
step_rewards = step_rewards[0]
print(step_rewards)



pygame.init()

screendim = (2000, 1200)
pygame.display.set_caption('PA')
window_surface = pygame.display.set_mode(screendim)

background = pygame.Surface(screendim)
background.fill(pygame.Color('#FFFFFF'))

is_running = True

counter = 0
while is_running:

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            is_running = False

    window_surface.blit(background, (0, 0))

    background.fill(pygame.Color('#FFFFFF'))
    draw_items(items, background, actions[:counter])
    draw_graph(items, nodes, coords, background, actions[:counter], step_rewards[:counter])
    pygame.display.update()
    time.sleep(0.1)
    counter += 1
