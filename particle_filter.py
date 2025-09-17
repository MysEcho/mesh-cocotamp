import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from matplotlib.widgets import Slider, Button
from matplotlib.patches import Wedge
import matplotlib.animation as animation


class ParticleFilter:
    def __init__(self, num_particles, box_center, box_size):
        self.num_particles = num_particles
        self.box_center = box_center
        self.box_size = box_size
        self.particles = np.zeros((self.num_particles, 2))  # Each particle is [x, y]
        self.weights = np.ones(self.num_particles, dtype=np.float64) / self.num_particles  # Ensure weights are float
        self.first_update = True  # Flag to track first update
        
        # Initialize particles randomly within the box
        self.particles[:, 0] = np.random.uniform(box_center[0] - box_size[0]/2, box_center[0] + box_size[0]/2, self.num_particles)
        self.particles[:, 1] = np.random.uniform(box_center[1] - box_size[1]/2, box_center[1] + box_size[1]/2, self.num_particles)
    
    def predict(self):
        """
        Predict the next state of the particles based on the motion model.
        motion is [delta_x, delta_y].
        """
        # Apply kernel density estimation (Gaussian kernel centered at each particle)
        cov_matrix = np.diag([0.05, 0.05]) ** 2  # Covariance matrix for Gaussian
        self.particles = np.array([np.random.multivariate_normal(mean, cov_matrix) for mean in self.particles])
    
    def update(self, measurement, measurement_noise, occ_obj_center, occ_obj_size):
        """
        Update the weights of each particle based on the observation function.
        measurement is [measured_x, measured_y].
        measurement_noise is the standard deviation of the measurement noise.
        """
        # Apply kernel density estimation (Gaussian kernel centered at each particle)
        cov_matrix = np.diag(measurement_noise) ** 2  # Covariance matrix for Gaussian
        new_particles = np.array([np.random.multivariate_normal(mean, cov_matrix) for mean in self.particles])
        new_weights = np.array([multivariate_normal.pdf(measurement, mean=particle, cov=cov_matrix) for particle in self.particles])
        
        if self.first_update:
            # On first update, filter out particles that are visible
            keep_indices = []
            for i, particle in enumerate(self.particles):
                # Keep particles that are not visible OR are occluded, AND are inside the box
                if (self.not_visible(particle) or self.is_occluded(particle, occ_obj_center, occ_obj_size)) and self.is_inside_box(particle):
                    new_weights[i] = self.weights[i]
                    keep_indices.append(i)
                else:
                    new_weights[i] = 1e-300
            
            self.particles = new_particles[keep_indices]
            self.weights = new_weights[keep_indices]
            self.num_particles = len(self.particles)
            self.first_update = False  # Set flag to False after first update
        else:
            # On subsequent updates, just update weights but don't filter
            for i, particle in enumerate(self.particles):
                if (self.not_visible(particle) or self.is_occluded(particle, occ_obj_center, occ_obj_size)) and self.is_inside_box(particle):
                    new_weights[i] = self.weights[i]
                else:
                    new_weights[i] = 1e-300
            
            self.particles = new_particles
            self.weights = new_weights
    
    def not_visible(self, particle):
        """
        Check if a particle is not in the visible region.
        """
        # Calculate the distance and angle from the robot to the particle
        dx = particle[0] - robot_pose[0]
        dy = particle[1] - robot_pose[1]
        distance = np.sqrt(dx**2 + dy**2)
        angle = np.arctan2(dy, dx) - robot_pose[2]

        # Check if the particle is not within the visibility distance and angle
        return distance > visibility_distance or angle < -theta or angle > theta
    
    def is_inside_box(self, particle):
        """
        Check if a particle is inside a given box.
        particle is [x, y].
        box_center is [center_x, center_y].
        box_size is [width, height].
        """
        # Calculate the box boundaries
        left_boundary = self.box_center[0] - self.box_size[0] / 2
        right_boundary = self.box_center[0] + self.box_size[0] / 2
        bottom_boundary = self.box_center[1] - self.box_size[1] / 2
        top_boundary = self.box_center[1] + self.box_size[1] / 2

        # Check if the particle is inside the box
        return left_boundary <= particle[0] <= right_boundary and \
            bottom_boundary <= particle[1] <= top_boundary
    
    def is_occluded(self, particle, occ_obj_center, occ_obj_size):
        """
        Check if a particle is occluded by the object using raycasting.
        """
        # Check if the particle is inside the occ_obj
        # Calculate the box boundaries
        left_boundary = occ_obj_center[0] - occ_obj_size[0] / 2
        right_boundary = occ_obj_center[0] + occ_obj_size[0] / 2
        bottom_boundary = occ_obj_center[1] - occ_obj_size[1] / 2
        top_boundary = occ_obj_center[1] + occ_obj_size[1] / 2

        # Check if the particle is inside the box
        if left_boundary <= particle[0] <= right_boundary and \
            bottom_boundary <= particle[1] <= top_boundary:
            return False
        
        # Define the start and end points of the ray
        start_point = robot_pose[:2]
        end_point = particle

        # Define the corners of the occluding object
        obj_corners = [
            occ_obj_center + np.array([-occ_obj_size[0]/2, -occ_obj_size[1]/2]),
            occ_obj_center + np.array([occ_obj_size[0]/2, -occ_obj_size[1]/2]),
            occ_obj_center + np.array([occ_obj_size[0]/2, occ_obj_size[1]/2]),
            occ_obj_center + np.array([-occ_obj_size[0]/2, occ_obj_size[1]/2])
        ]

        # Check if the ray intersects with any of the edges of the occluding object
        for i in range(len(obj_corners)):
            p1 = obj_corners[i]
            p2 = obj_corners[(i + 1) % len(obj_corners)]
            if self.ray_intersects_segment(start_point, end_point, p1, p2):
                return True
        return False

    def ray_intersects_segment(self, p1, p2, q1, q2):
        """
        Check if the ray from p1 to p2 intersects with the segment from q1 to q2.
        """
        def ccw(A, B, C):
            return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

        return ccw(p1, q1, q2) != ccw(p2, q1, q2) and ccw(p1, p2, q1) != ccw(p1, p2, q2)
        
    def normalize_weights(self):
        """
        Normalize the weights of the particles.
        """
        self.weights /= np.sum(self.weights)
    
    def resample(self):
        """
        Resample particles based on their weights.
        """
        indices = np.random.choice(range(len(self.particles)), size=len(self.particles), p=self.weights/np.sum(self.weights))
        self.particles = self.particles[indices]
        self.weights.fill(1.0 / len(self.particles))
    
    def estimate(self):
        """
        Estimate the pose of the object as the weighted mean of the particles.
        """
        mean = np.average(self.particles, weights=self.weights, axis=0)
        return mean

# Initialize the Particle Filter
num_particles = 1000
box_center = (2.5, 2.5)
box_size = (2.5, 1.5) # width, height
occ_obj_center = np.array([3, 2])
occ_obj_size = np.array([1, 0.25]) # width, height
pf = ParticleFilter(num_particles, box_center, box_size)

# Define the robot's pose
robot_pose = np.array([2.5, 1, np.pi/2])  # [x, y, theta]

# Define the visibility angle and distance
theta = np.pi / 5  # 45 degrees
visibility_distance = 3  # Adjust as needed

# Assume the true pose of the object is at (3, 3)
true_pose = np.array([3, 2.5])
measurement_noise = [0.0001, 0.0001]

# Create figure for plotting
fig, ax = plt.subplots(figsize=(5, 5))
plt.subplots_adjust(bottom=0.35)

# Define the update function for the animation
def update_plot(frame):
    # Clear the previous plot
    ax.clear()
    
    # Plot the initial state before any updates
    if frame == 0:
        ax.scatter(pf.particles[:, 0], pf.particles[:, 1], color='r', s=1, label='Particles')
        ax.scatter(true_pose[0], true_pose[1], color='g', label='True Pose')
        # Step 5: Estimate the pose
        estimated_pose = pf.estimate()
    else:
        # Simulate a measurement with some noise
        measurement = true_pose + np.random.normal(0, measurement_noise, size=2)
        
        # Step 1: Predict
        pf.predict()
        
        # Step 2: Update
        pf.update(measurement, measurement_noise, occ_obj_center, occ_obj_size)
        
        # Step 3: Normalize weights
        pf.normalize_weights()
        
        # Step 4: Resample
        pf.resample()
        
        # Step 5: Estimate the pose
        estimated_pose = pf.estimate()
        print(pf.num_particles)

        # Plot the particles and the estimated pose
        ax.scatter(pf.particles[:, 0], pf.particles[:, 1], color='r', s=1, label='Particles')
        ax.scatter(true_pose[0], true_pose[1], color='g', label='True Pose')
        ax.scatter(estimated_pose[0], estimated_pose[1], color='b', label='Estimated Pose')
    
    # Draw a box around the true pose
    box = plt.Rectangle((box_center[0] - box_size[0]/2, box_center[1] - box_size[1]/2), box_size[0], box_size[1], fill=False, color='blue', label='Initial Box')
    ax.add_patch(box)

    # Draw a circle representing a robot
    robot = plt.Circle((robot_pose[0], robot_pose[1]), 0.05, color='purple')
    ax.add_patch(robot)

    # Draw a circular sector representing the robot's visibility region
    visibility_region = Wedge(center=robot_pose[:2], r=visibility_distance, theta1=np.degrees(robot_pose[2]-theta), theta2=np.degrees(robot_pose[2]+theta), color='yellow', alpha=0.5)
    ax.add_patch(visibility_region)

    # Draw a wide rectangle that occludes the object
    # occluding_obj = plt.Rectangle((2.5, 2), 1, 0.25, fill=True, color='gray', alpha=0.5, label='Occlusion')
    occluding_obj = plt.Rectangle((occ_obj_center[0] - occ_obj_size[0]/2, 
                                   occ_obj_center[1] - occ_obj_size[1]/2), 
                                   occ_obj_size[0], occ_obj_size[1], fill=True, color='gray', alpha=0.5, label='Occlusion')
    ax.add_patch(occluding_obj)

    ax.legend()
    ax.set_xlim(0, 5)
    ax.set_ylim(0, 5)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title(f"Iteration {frame}: Estimated Pose {estimated_pose}")
    
# Create slider
ax_slider = plt.axes([0.1, 0.2, 0.8, 0.03], facecolor='lightgoldenrodyellow')
slider = Slider(ax_slider, 'Frame', 0, 9, valinit=0, valstep=1)

# Create play button
ax_button = plt.axes([0.45, 0.05, 0.1, 0.04])
button = Button(ax_button, 'Play')

def update(val):
    frame = int(slider.val)
    update_plot(frame)
    fig.canvas.draw_idle()

slider.on_changed(update)

# Animation play function
playing = False

def play(event):
    global playing
    playing = not playing
    if playing:
        button.label.set_text('Pause')
    else:
        button.label.set_text('Play')
    for frame in range(slider.val, 10):
        if not playing:
            break
        slider.set_val(frame)
        plt.pause(0.5)  # Adjust the pause duration as needed

button.on_clicked(play)

# Initial plot
update_plot(0)

# Show the plot with slider and play button
plt.show()

# Create the animation
ani = animation.FuncAnimation(fig, update_plot, frames=10, repeat=True)

# Save the animation as a gif
ani.save('particle_filter_occ.gif', writer='imagemagick', fps=2)