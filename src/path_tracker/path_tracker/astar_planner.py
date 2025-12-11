#!/usr/bin/env python3
"""
A* Path Planning Algorithm for Collision-Free Navigation

This module implements the A* algorithm for path planning on occupancy grid maps.
It finds the shortest collision-free path from start to goal position.

Author: CRAIP Team
"""

import math
import heapq
from typing import List, Tuple, Optional, Set
import numpy as np


class Node:
    """
    Represents a node in the A* search graph.
    """
    def __init__(self, x: int, y: int, g_cost: float = float('inf'), h_cost: float = 0.0, parent=None):
        self.x = x
        self.y = y
        self.g_cost = g_cost  # Cost from start to this node
        self.h_cost = h_cost  # Heuristic cost from this node to goal
        self.f_cost = g_cost + h_cost  # Total cost
        self.parent = parent  # Parent node for path reconstruction
    
    def __lt__(self, other):
        """For priority queue comparison"""
        if self.f_cost != other.f_cost:
            return self.f_cost < other.f_cost
        return self.h_cost < other.h_cost
    
    def __eq__(self, other):
        """For set operations"""
        return self.x == other.x and self.y == other.y
    
    def __hash__(self):
        """For set operations"""
        return hash((self.x, self.y))
    
    def __repr__(self):
        return f"Node({self.x}, {self.y}, f={self.f_cost:.2f})"


class AStarPlanner:
    """
    A* path planner for occupancy grid maps.
    """
    
    def __init__(self, inflation_radius: float = 0.3):
        """
        Initialize A* planner.
        
        Args:
            inflation_radius: Robot radius in meters for obstacle inflation
        """
        self.inflation_radius = inflation_radius
        self.map_data = None
        self.map_metadata = None
        
    def set_map(self, occupancy_grid, map_metadata):
        """
        Set the occupancy grid map for planning.
        
        Args:
            occupancy_grid: 2D numpy array of occupancy values (0-100)
            map_metadata: Dictionary with 'resolution', 'origin_x', 'origin_y', 'width', 'height'
        """
        self.map_data = occupancy_grid
        self.map_metadata = map_metadata
        
    def world_to_map(self, world_x: float, world_y: float) -> Tuple[int, int]:
        """
        Convert world coordinates to map cell coordinates.
        
        Args:
            world_x: X coordinate in world frame (meters)
            world_y: Y coordinate in world frame (meters)
            
        Returns:
            Tuple of (map_x, map_y) cell indices
        """
        if self.map_metadata is None:
            raise ValueError("Map not set. Call set_map() first.")
        
        resolution = self.map_metadata['resolution']
        origin_x = self.map_metadata['origin_x']
        origin_y = self.map_metadata['origin_y']
        
        map_x = int((world_x - origin_x) / resolution)
        map_y = int((world_y - origin_y) / resolution)
        
        return map_x, map_y
    
    def map_to_world(self, map_x: int, map_y: int) -> Tuple[float, float]:
        """
        Convert map cell coordinates to world coordinates.
        
        Args:
            map_x: X cell index
            map_y: Y cell index
            
        Returns:
            Tuple of (world_x, world_y) in meters
        """
        if self.map_metadata is None:
            raise ValueError("Map not set. Call set_map() first.")
        
        resolution = self.map_metadata['resolution']
        origin_x = self.map_metadata['origin_x']
        origin_y = self.map_metadata['origin_y']
        
        world_x = map_x * resolution + origin_x
        world_y = map_y * resolution + origin_y
        
        return world_x, world_y
    
    def is_valid_cell(self, map_x: int, map_y: int) -> bool:
        """
        Check if a map cell is valid (within bounds and not occupied).
        
        Args:
            map_x: X cell index
            map_y: Y cell index
            
        Returns:
            True if cell is valid for path planning
        """
        if self.map_data is None:
            return False
        
        height, width = self.map_data.shape
        
        # Check bounds
        if map_x < 0 or map_x >= width or map_y < 0 or map_y >= height:
            return False
        
        # Check if cell is occupied (value > 50 means occupied)
        # Also check inflation radius around the cell
        cell_value = self.map_data[map_y, map_x]
        
        if cell_value > 50:  # Occupied
            return False
        
        # Inflate obstacles by checking nearby cells
        inflation_cells = int(self.inflation_radius / self.map_metadata['resolution'])
        for dy in range(-inflation_cells, inflation_cells + 1):
            for dx in range(-inflation_cells, inflation_cells + 1):
                check_x = map_x + dx
                check_y = map_y + dy
                
                if (check_x >= 0 and check_x < width and 
                    check_y >= 0 and check_y < height):
                    dist = math.sqrt(dx*dx + dy*dy) * self.map_metadata['resolution']
                    if dist <= self.inflation_radius:
                        if self.map_data[check_y, check_x] > 50:
                            return False
        
        return True
    
    def heuristic(self, node1: Node, node2: Node) -> float:
        """
        Calculate heuristic cost (Euclidean distance) between two nodes.
        
        Args:
            node1: First node
            node2: Second node
            
        Returns:
            Heuristic cost in meters
        """
        dx = node1.x - node2.x
        dy = node1.y - node2.y
        distance = math.sqrt(dx*dx + dy*dy) * self.map_metadata['resolution']
        return distance
    
    def get_neighbors(self, node: Node) -> List[Node]:
        """
        Get valid neighboring nodes (8-connected).
        
        Args:
            node: Current node
            
        Returns:
            List of valid neighbor nodes
        """
        neighbors = []
        
        # 8-connected neighbors
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                
                new_x = node.x + dx
                new_y = node.y + dy
                
                if self.is_valid_cell(new_x, new_y):
                    neighbors.append(Node(new_x, new_y))
        
        return neighbors
    
    def get_move_cost(self, node1: Node, node2: Node) -> float:
        """
        Calculate movement cost between two adjacent nodes.
        
        Args:
            node1: First node
            node2: Second node
            
        Returns:
            Movement cost in meters
        """
        dx = node2.x - node1.x
        dy = node2.y - node1.y
        
        # Diagonal moves cost more
        if abs(dx) == 1 and abs(dy) == 1:
            return math.sqrt(2) * self.map_metadata['resolution']
        else:
            return self.map_metadata['resolution']
    
    def plan_path(self, start_x: float, start_y: float, 
                  goal_x: float, goal_y: float) -> Optional[List[Tuple[float, float]]]:
        """
        Plan a path from start to goal using A* algorithm.
        
        Args:
            start_x: Start X coordinate in world frame (meters)
            start_y: Start Y coordinate in world frame (meters)
            goal_x: Goal X coordinate in world frame (meters)
            goal_y: Goal Y coordinate in world frame (meters)
            
        Returns:
            List of (x, y) waypoints in world coordinates, or None if no path found
        """
        if self.map_data is None or self.map_metadata is None:
            raise ValueError("Map not set. Call set_map() first.")
        
        # Convert to map coordinates
        start_map_x, start_map_y = self.world_to_map(start_x, start_y)
        goal_map_x, goal_map_y = self.world_to_map(goal_x, goal_y)
        
        # Validate start and goal
        if not self.is_valid_cell(start_map_x, start_map_y):
            print(f"Warning: Start position ({start_x}, {start_y}) is in an occupied cell")
            return None
        
        if not self.is_valid_cell(goal_map_x, goal_map_y):
            print(f"Warning: Goal position ({goal_x}, {goal_y}) is in an occupied cell")
            return None
        
        # Initialize start and goal nodes
        start_node = Node(start_map_x, start_map_y, g_cost=0.0)
        goal_node = Node(goal_map_x, goal_map_y)
        
        start_node.h_cost = self.heuristic(start_node, goal_node)
        start_node.f_cost = start_node.g_cost + start_node.h_cost
        
        # Open set (priority queue) and closed set
        open_set = [start_node]
        heapq.heapify(open_set)
        closed_set: Set[Node] = set()
        
        # Dictionary to track best g_cost for each node
        g_costs = {(start_node.x, start_node.y): 0.0}
        
        # A* search
        while open_set:
            # Get node with lowest f_cost
            current = heapq.heappop(open_set)
            
            # Skip if already processed with better cost
            if (current.x, current.y) in closed_set:
                continue
            
            closed_set.add((current.x, current.y))
            
            # Check if goal reached
            if current.x == goal_node.x and current.y == goal_node.y:
                # Reconstruct path
                path = []
                node = current
                while node is not None:
                    world_x, world_y = self.map_to_world(node.x, node.y)
                    path.append((world_x, world_y))
                    node = node.parent
                path.reverse()
                return path
            
            # Explore neighbors
            for neighbor in self.get_neighbors(current):
                neighbor_key = (neighbor.x, neighbor.y)
                
                # Skip if already in closed set
                if neighbor_key in closed_set:
                    continue
                
                # Calculate tentative g_cost
                move_cost = self.get_move_cost(current, neighbor)
                tentative_g = current.g_cost + move_cost
                
                # Check if this is a better path
                if neighbor_key not in g_costs or tentative_g < g_costs[neighbor_key]:
                    neighbor.g_cost = tentative_g
                    neighbor.h_cost = self.heuristic(neighbor, goal_node)
                    neighbor.f_cost = neighbor.g_cost + neighbor.h_cost
                    neighbor.parent = current
                    
                    g_costs[neighbor_key] = tentative_g
                    heapq.heappush(open_set, neighbor)
        
        # No path found
        print("A*: No path found from start to goal")
        return None
    
    def simplify_path(self, path: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        """
        Simplify path by removing unnecessary waypoints (line-of-sight check).
        
        Args:
            path: Original path as list of (x, y) tuples
            
        Returns:
            Simplified path
        """
        if len(path) <= 2:
            return path
        
        simplified = [path[0]]
        i = 0
        
        while i < len(path) - 1:
            # Try to skip as many points as possible
            for j in range(len(path) - 1, i + 1, -1):
                if self.has_line_of_sight(path[i], path[j]):
                    simplified.append(path[j])
                    i = j
                    break
            else:
                # No line of sight, add next point
                i += 1
                if i < len(path):
                    simplified.append(path[i])
        
        return simplified
    
    def has_line_of_sight(self, point1: Tuple[float, float], 
                          point2: Tuple[float, float]) -> bool:
        """
        Check if there's a clear line of sight between two points.
        
        Args:
            point1: First point (x, y)
            point2: Second point (x, y)
            
        Returns:
            True if line of sight is clear
        """
        x1, y1 = point1
        x2, y2 = point2
        
        # Convert to map coordinates
        map_x1, map_y1 = self.world_to_map(x1, y1)
        map_x2, map_y2 = self.world_to_map(x2, y2)
        
        # Bresenham's line algorithm to check all cells along the line
        dx = abs(map_x2 - map_x1)
        dy = abs(map_y2 - map_y1)
        sx = 1 if map_x1 < map_x2 else -1
        sy = 1 if map_y1 < map_y2 else -1
        err = dx - dy
        
        x, y = map_x1, map_y1
        
        while True:
            if not self.is_valid_cell(x, y):
                return False
            
            if x == map_x2 and y == map_y2:
                break
            
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x += sx
            if e2 < dx:
                err += dx
                y += sy
        
        return True

