import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle
import numpy as np

def create_gif(best_coordinates: list[list[float]], best_radii: list[float], N: int, square_size: float, output_file: str = "animation.gif"):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(0, square_size)
    ax.set_ylim(0, square_size)
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True)

    # Initialize circles
    circles = [Circle((0, 0), 0, fill=False, edgecolor='green') for _ in range(N)]
    for circle in circles:
        ax.add_patch(circle)

    # Update function for each frame
    def update(frame: int):
        positions = best_coordinates[frame]  # Extract positions for this frame
        radius = best_radii[frame]  # Extract radius for this frame
        for i, circle in enumerate(circles):
            circle.center = positions[i]
            circle.radius = radius
        ax.set_title(f"Iteration {frame + 1} / {len(best_coordinates)}")
        if frame % 100 == 0:
            print(f"Frame {frame + 1}")
        return circles

    # Create the animation
    animation = FuncAnimation(fig, update, frames=len(best_coordinates), interval=100, blit=True)

    # Save the animation as a GIF
    animation.save(output_file, writer="pillow")
    print(f"GIF saved as {output_file}")