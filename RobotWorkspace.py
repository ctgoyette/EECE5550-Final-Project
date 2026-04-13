import matplotlib.pyplot as plt
from matplotlib.markers import MarkerStyle
import numpy as np
import heapq


class RobotWorkspace:
    def __init__(self):
        self.X, self.Y, self.x0, self.y0 = self.initialize_workspace()
        self.nodes = set(zip(self.X.flatten(), self.Y.flatten()))
        self.point_coords = {'x': self.x0, 'y': self.y0}
        self.goal_coords = {'x': None, 'y': None}
        self.block_coords = {'x': None, 'y': None}
        self.step = 5
        plt.style.use('dark_background')
        
        self.fig, self.ax = plt.subplots()
        
        self.ax.plot(self.X, self.Y, 's', markersize=10, label='Workspace Points', color='white')
        m = MarkerStyle(marker='1')
        self.path_block_line, = self.ax.plot([], [], color='yellow', linestyle='--', linewidth=5)
        self.path_goal_line, = self.ax.plot([], [], color='yellow', linestyle='--', linewidth=5)
        self.point, = self.ax.plot(self.point_coords['x'], self.point_coords['y'], marker=m, markersize=40, color='red')
        self.block, = self.ax.plot([], [], marker='o', markersize=15, color='blue')
        self.goal, = self.ax.plot([], [], marker='*', markersize=30, color='green')
 
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_title('Robot Workspace')
        self.ax.grid(False)
        self.ax.axis('equal')
        fig_manager = plt.get_current_fig_manager()
        fig_manager.window.setGeometry(0, 0, 750, 750)
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)
    
    def initialize_workspace(self):
        x_min, x_max = 0, 115
        y_min, y_max = 0, 115
        
        x = np.linspace(x_min, x_max, 24)
        y = np.linspace(y_min, y_max, 24)
        X, Y = np.meshgrid(x, y)
        x0, y0 = x[3], y[3]  
        
        return X, Y, x0, y0
    
    def set_block(self, x, y):
        self.block_coords['x'] = x
        self.block_coords['y'] = y
        self.block.set_data([self.block_coords['x']], [self.block_coords['y']])
        self.fig.canvas.draw()

    def set_goal(self, x, y):
        self.goal_coords['x'] = x
        self.goal_coords['y'] = y
        self.goal.set_data([self.goal_coords['x']], [self.goal_coords['y']])
        self.fig.canvas.draw()

    def set_obstacle(self, x, y):
        # For group of points

        self.ax.plot(x, y, marker='X', markersize=20, color='magenta', label='Obstacle')
        self.nodes.discard((x, y))  # Remove the obstacle point from the set of nodes
        self.fig.canvas.draw()

    def update_robot(self, x, y):
        self.point_coords['x'] = x
        self.point_coords['y'] = y
        self.point.set_data([self.point_coords['x']], [self.point_coords['y']])
        self.fig.canvas.draw()

    def update_block(self, x, y):
        self.block_coords['x'] = x
        self.block_coords['y'] = y
        self.block.set_data([self.block_coords['x']], [self.block_coords['y']])
        self.fig.canvas.draw()

    def update_goal(self, x, y):
        self.goal_coords['x'] = x
        self.goal_coords['y'] = y
        self.goal.set_data([self.goal_coords['x']], [self.goal_coords['y']])
        self.fig.canvas.draw()
            
    def get_neighbors(self, node):
        x, y = node
        neighbors = []
        directions = [(0, 5), (0, -5), (5, 0), (-5, 0), (5, 5), (5, -5), (-5, 5), (-5, -5)]
        
        for dx, dy in directions:
            neighbor = (x + dx, y + dy)
            if neighbor in self.nodes:
                cost = np.sqrt(dx**2 + dy**2)
                neighbors.append((neighbor, cost))
        return neighbors

    def find_path(self, start, goal, line_obj):
        point1 = (start[0], start[1])
        point2 = (goal[0], goal[1])

        pq = [(0, point1)]
        g_score = {point1: 0}
        came_from = {}

        while pq:
            curr_dist, current = heapq.heappop(pq)
            if current == point2:
                path = self.draw_path(came_from, current, line_obj)
                return path

            if curr_dist > g_score.get(current, float('inf')):
                continue

            for neighbor, weight in self.get_neighbors(current):
                new_cost = curr_dist + weight
                if new_cost < g_score.get(neighbor, float('inf')):
                    g_score[neighbor] = new_cost
                    came_from[neighbor] = current
                    heapq.heappush(pq, (new_cost, neighbor))

    def draw_path(self, came_from, current, line_obj):
        path_x, path_y = [], []
        
        while current in came_from:
            path_x.append(current[0])
            path_y.append(current[1])
            current = came_from[current]
            
        path_x.append(current[0])
        path_y.append(current[1])
        path_x.reverse()
        path_y.reverse()

        line_obj.set_data(path_x, path_y)
        self.fig.canvas.draw()

        return list(zip(path_x, path_y))
    
    def on_key(self, event):
        
        m = MarkerStyle('1')
        angle = 0

        if event.key == 'u':
            self.point_coords['y'] += self.step
            angle = 0
        elif event.key == 'j':
            self.point_coords['y'] -= self.step
            angle = 180
        elif event.key == 'h':
            self.point_coords['x'] -= self.step
            angle = 90
        elif event.key == 'k':
            self.point_coords['x'] += self.step
            angle = 270
        elif event.key == 'y':
            self.point_coords['x'] -= self.step
            self.point_coords['y'] += self.step
            angle = 45
        elif event.key == 'i':
            self.point_coords['x'] += self.step
            self.point_coords['y'] += self.step
            angle = 315
        elif event.key == 'n':
            self.point_coords['x'] -= self.step
            self.point_coords['y'] -= self.step
            angle = 135
        elif event.key == 'm':
            self.point_coords['x'] += self.step
            self.point_coords['y'] -= self.step
            angle = 225
        else:
            return 

        m._transform.rotate_deg(angle) 
        
        self.point.set_marker(m)
        self.point.set_data([self.point_coords['x']], [self.point_coords['y']])
        self.fig.canvas.draw()

if __name__ == "__main__":
    workspace = RobotWorkspace()
    block_x, block_y = [20, 80]  # Setting block position (can be changed as needed)
    goal_x, goal_y = [90, 90]   # Setting goal position (can be changed as needed)
    workspace.set_block(block_x, block_y) # Place the block in the workspace
    workspace.set_goal(goal_x, goal_y)   # Place the goal in the workspace
    workspace.set_obstacle([15, 25], [25, 30])  # Place an obstacle in the workspace
    # Find path from start to block, then from block to goal
    path = workspace.find_path((workspace.x0, workspace.y0), (block_x, block_y), workspace.path_block_line)
    path2 = workspace.find_path((block_x, block_y), (goal_x, goal_y), workspace.path_goal_line)
    path2.pop(0)  # Remove the block position from the second path to avoid duplication
    path.extend(path2)  # Combine paths for full route
    # For Control, use the path variable which contains the list of coordinates from start to block and then block to goal
    # List of Positions from initial to block, then block to goal
    print(f"{'Step':<6} | {'Coordinate':<15}")
    print("-" * 25)
    for i, (x, y) in enumerate(path):
        print(f"{i+1:<6} | ({x:>5}, {y:>5})")
    for i, (x, y) in enumerate(path2):
        print(f"{i+1+len(path):<6} | ({x:>5}, {y:>5})")
    plt.show()
    



