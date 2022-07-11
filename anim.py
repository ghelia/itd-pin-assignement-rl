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
SIZE = 80

pygame.init()
screendim = (2000, 1200)
pygame.display.set_caption('PA')
window_surface = pygame.display.set_mode(screendim)
background = pygame.Surface(screendim)
background.fill(pygame.Color('#FFFFFF'))

colors = [
    np.array((0,155,0)),
    np.array((30,30,255)),
    np.array((155,155,155)),
    np.array((0,255,255)),
    np.array((155,255,0))
]


def draw_polygon_alpha(surface, color, points):
    lx, ly = zip(*points)
    min_x, min_y, max_x, max_y = min(lx), min(ly), max(lx), max(ly)
    target_rect = pygame.Rect(min_x, min_y, max_x - min_x, max_y - min_y)
    shape_surf = pygame.Surface(target_rect.size, pygame.SRCALPHA)
    pygame.draw.polygon(shape_surf, color, [(x - min_x, y - min_y) for x, y in points])
    surface.blit(shape_surf, target_rect)

def items_attentions(attentions):
    half = SIZE/3
    factor = 200/attentions.max()
    for idx,pos in enumerate(items_positions()):
        score = attentions[idx] * factor
        color = (255, 0, 0, score)
        draw_polygon_alpha(
            background,
            color,
            (
                (pos[0] - half, pos[1] - half),
                (pos[0] - half, pos[1] + half),
                (pos[0] + half, pos[1] + half),
                (pos[0] + half, pos[1] - half)
            )
        )

def nodes_attentions(coords, attentions):
    half = SIZE/2
    factor = 200/attentions.max()
    for idx,pos in enumerate(nodes_positions(coords)):
        score = attentions[idx] * factor
        color = (255, 0, 0, score)
        draw_polygon_alpha(
            background,
            color,
            (
                (pos[0] - half, pos[1] - half),
                (pos[0] - half, pos[1] + half),
                (pos[0] + half, pos[1] + half),
                (pos[0] + half, pos[1] - half)
            )
        )

def highlight(pos):
    half = SIZE/2
    pygame.draw.polygon(
        background,
        (255,0,0),
        (
            (pos[0] - half, pos[1] - half),
            (pos[0] - half, pos[1] + half),
            (pos[0] + half, pos[1] + half),
            (pos[0] + half, pos[1] - half)
        ),
        width=3
    )

def distance(pos):
    mouse = pygame.mouse.get_pos()
    pos = (pos[0] - mouse[0], pos[1] - mouse[1])
    return np.sqrt(pos[0] ** 2 + pos[1] ** 2)

def mouse_over_item():
    for idx,pos in enumerate(items_positions()):
        if distance(pos) < SIZE/2:
            return idx
    return -1

def mouse_over_node(coords):
    for idx,pos in enumerate(nodes_positions(coords)):
        if distance(pos) < SIZE/2:
            return idx
    return -1

def items_positions():
    positions = []
    line_length = 7
    startx = 30
    starty = 30
    x = 0
    y = 0
    wgap = 55
    hgap = 70
    for idx in range(Config.nitems):
        pos = (startx + x * wgap, starty + y * hgap)
        positions.append(pos)
        x += 1
        if x >= line_length:
            x = 0
            y += 1
    return positions

def nodes_positions(coords):
    positions = []
    center = (1200, 600)
    size = SIZE
    for idx in range(Config.nitems):
        node = nodes[idx]
        coord = coords[idx]
        pos = (
            center[0] + size * coord[0],
            center[1] + size * coord[1]
        )
        positions.append(pos)
    return positions


def draw_item(item, pos):
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


def draw_items(items, actions):
    selected = [action[0] for action in actions]
    for idx, item in enumerate(items):
        pos = items_positions()[idx]
        if idx not in selected:
            draw_item(item, pos)


def draw_node(node, pos):
    border = 3
    half = int(SIZE/2)
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

def draw_error(pos):
    half = int(SIZE/2)
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

def draw_graph(items, nodes, coords, actions, step_rewards):
    table = {int(action[1]):int(action[0]) for action in actions}
    steps = {idx:int(action[1]) for idx,action in enumerate(actions)}
    for idx in range(len(nodes)):
        node = nodes[idx]
        reward = 0
        if idx < len(step_rewards):
            reward = step_rewards[idx]
        pos = nodes_positions(coords)[idx]
        draw_node(node, pos)
        if idx in table.keys():
            draw_item(items[table[idx]], pos)

    for step,reward in enumerate(step_rewards):
        if reward < 0:
            idx = steps[step]
            pos = nodes_positions(coords)[idx]
            draw_error(pos)


agent = Agent()
agent.train()
agent.load_state_dict(torch.load("./agent-5261-increase-size.chkpt"))
items, nodes, edges, coords = batch(npins=Config.nitems, batch_size=Config.batch_size, return_coords=True)
with torch.no_grad():
    _, _, _, _, actions, rewards, step_rewards, attention_view = agent(items, nodes, edges, greedy=True, return_info=True)


items = items[0]
nodes = nodes[0]
edges = edges[0]
coords = coords[0]
actions = actions[0]
actions = [(actions[idx*2], actions[idx*2 + 1]) for idx in range(len(actions) // 2)]
rewards = rewards[0]
step_rewards = step_rewards[0]
print(step_rewards)





is_running = True

counter = 0
while is_running:

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            is_running = False
        if event.type == pygame.KEYDOWN:
            counter += 1
            if counter >= len(actions):
                counter = 0

    window_surface.blit(background, (0, 0))

    background.fill(pygame.Color('#FFFFFF'))
    draw_items(items, actions[:counter])
    draw_graph(items, nodes, coords, actions[:counter], step_rewards[:counter])
    if mouse_over_item() >= 0:
        highlight(items_positions()[mouse_over_item()])
        items_attentions(
            attention_view[counter]["item_encoder_1"][0][mouse_over_item()]
        )
    if mouse_over_node(coords) >= 0:
        highlight(nodes_positions(coords)[mouse_over_node(coords)])
        nodes_attentions(
            coords,
            attention_view[counter]["node_encoder_2"][0][mouse_over_node(coords)]
        )
    pygame.display.update()
    # time.sleep(0.1)
