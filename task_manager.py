import time
import math

class Task:
    def __init__(self, name, duration, target_pos, dependencies=None):
        self.name = name
        self.duration = duration
        self.target_pos = target_pos # Coordinates for the work step
        self.dependencies = dependencies if dependencies else []
        
        self.status = "PENDING"  # PENDING, IN_PROGRESS, COMPLETED
        self.assigned_robot = None
        self.completion_time = None

    def is_ready(self, completed_tasks):
        """Check if all dependencies are met."""
        for dep in self.dependencies:
            if dep not in completed_tasks:
                return False
        return True

    def complete(self):
        self.status = "COMPLETED"
        self.completion_time = time.time()

class TaskManager:
    def __init__(self):
        self.tasks = []
        self.completed_tasks = []
        self.log = []

    def add_task(self, task):
        self.tasks.append(task)

    def get_next_available_tasks(self):
        """Return tasks that are PENDING and have all dependencies met."""
        available = []
        for task in self.tasks:
            if task.status == "PENDING" and task.is_ready(self.completed_tasks):
                available.append(task)
        return available

    def update(self):
        """Update the list of completed tasks."""
        for task in self.tasks:
            if task.status == "COMPLETED" and task.name not in self.completed_tasks:
                self.completed_tasks.append(task.name)
                self.log.append({
                    "task": task.name,
                    "time": time.strftime("%H:%M:%S"),
                    "robot": task.assigned_robot
                })

    def is_all_completed(self):
        return len(self.completed_tasks) == len(self.tasks)

    def schedule_tasks(self, robots):
        """
        AI-driven scheduling (Greedy/Proximity-based).
        Assigns available tasks to idle robots.
        """
        available_tasks = self.get_next_available_tasks()
        idle_robots = [r for r in robots if r.status == "IDLE"]
        
        # Simple optimization: assign nearest task to idle robot
        for robot in idle_robots:
            if not available_tasks:
                break
            
            # Find closest task
            best_task = None
            min_dist = float('inf')
            
            for task in available_tasks:
                # Calculate distance between robot home and task target
                dist = math.sqrt(sum((a - b)**2 for a, b in zip(robot.home_pos, task.target_pos)))
                if dist < min_dist:
                    min_dist = dist
                    best_task = task
            
            if best_task:
                best_task.status = "IN_PROGRESS"
                best_task.assigned_robot = robot.name
                robot.assign_task(best_task)
                available_tasks.remove(best_task)
