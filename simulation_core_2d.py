import numpy as np
import matplotlib.pyplot as plt
import time

class WarehouseSimulation2D:
    def __init__(self, width=10, height=10):
        self.width = width
        self.height = height
        self.amr_pos = np.array([2.0, 2.0])
        self.target_pos = np.array([8.0, 8.0])
        self.shelves = [[4, 4], [6, 4], [4, 6], [6, 6]] # Obstacles

    def step(self, action):
        # Action: [v_x, v_y]
        self.amr_pos += action * 0.5
        # Boundaries
        self.amr_pos = np.clip(self.amr_pos, 0, self.width)
        
        # Collision with shelves (simplified)
        for shelf in self.shelves:
            if np.linalg.norm(self.amr_pos - shelf) < 0.5:
                self.amr_pos -= action * 0.5 # Bounce back
                
    def get_observation(self):
        return np.concatenate([self.amr_pos, self.target_pos])

    def render(self):
        plt.clf()
        plt.xlim(0, self.width)
        plt.ylim(0, self.height)
        # Draw Shelves
        for shelf in self.shelves:
            plt.gca().add_patch(plt.Rectangle((shelf[0]-0.4, shelf[1]-0.4), 0.8, 0.8, color='gray'))
        # Draw AMR
        plt.scatter(self.amr_pos[0], self.amr_pos[1], color='red', label='AMR')
        # Draw Target
        plt.scatter(self.target_pos[0], self.target_pos[1], color='green', marker='X', label='Target')
        plt.title("Warehouse Logistics 2D Simulation")
        plt.legend()
        plt.pause(0.01)

if __name__ == "__main__":
    sim = WarehouseSimulation2D()
    for _ in range(50):
        sim.step(np.random.uniform(-1, 1, 2))
        sim.render()
