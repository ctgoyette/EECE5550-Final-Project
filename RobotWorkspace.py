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
        m = MarkerStyle(marker='d')
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
        # fig_manager.window.setGeometry(0, 0, 750, 750)
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)
    
    def initialize_workspace(self):
        x_min, x_max = 0, 220
        y_min, y_max = 0, 220
        
        x = np.linspace(x_min, x_max, 23)
        y = np.linspace(y_min, y_max, 23)
        print(f"Workspace initialized: {x}, {y}")
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

        self.ax.plot(x, y, marker='X', markersize=15, color='orange', label='Obstacle')
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

    def get_obstacle_matrix(self):
        # Get all points that are NOT in the traversable nodes set
        all_pts = np.array(list(zip(self.X.flatten(), self.Y.flatten())))
        # Filter for nodes you have 'discarded' or the boundary
        is_obstacle = np.array([tuple(p) not in self.nodes for p in all_pts])
        return all_pts[is_obstacle]
    
    def get_virtual_lidar_array(self, robot_x, robot_y, robot_theta):
        obstacles = self.get_obstacle_matrix()
        if len(obstacles) == 0:
            return np.array([])

        # 1. Vectorized translation
        dx = obstacles[:, 0] - robot_x
        dy = obstacles[:, 1] - robot_y
        
        # 2. Polar conversion
        dist = np.sqrt(dx**2 + dy**2)
        # Global angle of the obstacle relative to robot
        angles = np.arctan2(dy, dx) - robot_theta
        
        # 3. Normalize to [-pi, pi]
        angles = (angles + np.pi) % (2 * np.pi) - np.pi
        
        # 4. Filter for Front 180 degrees (-90 to +90 degrees)
        mask = (angles >= -np.pi/2) & (angles <= np.pi/2)
        
        # Return as [angle, distance] sorted by angle
        virtual_scan = np.column_stack((angles[mask], dist[mask]))
        return virtual_scan[virtual_scan[:, 0].argsort()]
    
    def update_robot_icp(self, x, y, theta, actual_lidar_data):
        # 1. Get what the robot SHOULD see based on the map
        virtual_array = self.get_virtual_lidar_array(x, y, theta)
        
        # 2. Run ICP to find the offset between virtual and actual
        # actual_lidar_data should also be filtered to 0-180
        delta_x, delta_y, delta_theta = self.run_icp(actual_lidar_data, virtual_array)
        
        # 3. Apply the correction to the pose
        corrected_x = x + delta_x
        corrected_y = y + delta_y
        
        # 4. Update the visual marker
        self.point_coords['x'] = corrected_x
        self.point_coords['y'] = corrected_y
        self.point.set_data([corrected_x], [corrected_y])
        self.fig.canvas.draw_idle() # Use draw_idle for better performance
            
    def get_neighbors(self, node):
        x, y = node
        neighbors = []
        directions = [(0, 10), (0, -10), (10, 0), (-10, 0), (10, 10), (10, -10), (-10, 10), (-10, -10)]
        
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
    block_x, block_y = [100, 200]  # Setting block position (can be changed as needed)
    goal_x, goal_y = [200, 220]   # Setting goal position (can be changed as needed)
    workspace.set_block(block_x, block_y) # Place the block in the workspace
    workspace.set_goal(goal_x, goal_y)   # Place the goal in the workspace
    for x in range(50, 200, 10):
        workspace.set_obstacle(200, x)  # Place a horizontal line of obstacles
        workspace.set_obstacle(190, x)  # Place a horizontal line of obstacles
        workspace.set_obstacle(210, x)  # Place a horizontal line of obstacles
    
    # Find path from start to block, then from block to goal
    path = workspace.find_path((workspace.x0, workspace.y0), (block_x, block_y), workspace.path_block_line)
    path2 = workspace.find_path((block_x, block_y), (goal_x, goal_y), workspace.path_goal_line)
    # path2.pop(0)  # Remove the block position from the second path to avoid duplication
    #path.extend(path2)  # Combine paths for full route
    # For Control, use the path variable which contains the list of coordinates from start to block and then block to goal
    # List of Positions from initial to block, then block to goal
    print("Path from start to block:", path)
    print("Path from block to goal:", path2)
    plt.show()
    


