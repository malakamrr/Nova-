import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

SAFE_DISTANCE = 0.5
PREDICTION_STEPS = 10
SPACE_SIZE = 10

sat_pos = np.array([-8.0, 0.0])
sat_vel = np.array([0.1, 0.05])

num_debris = 5
np.random.seed(42)
debris_pos = np.random.uniform(-5, 5, (num_debris, 2))
debris_vel = np.random.uniform(-0.05, 0.05, (num_debris, 2))

def predict_and_avoid(sat_p, sat_v, deb_p, deb_v):
    global sat_vel
    warning = False
    
    for step in range(1, PREDICTION_STEPS + 1):
        future_sat = sat_p + (sat_v * step)
        future_deb = deb_p + (deb_v * step)
        
        distances = np.linalg.norm(future_deb - future_sat, axis=1)
        
        if np.any(distances < SAFE_DISTANCE):
            warning = True
            closest_debris_idx = np.argmin(distances)
            escape_vector = future_sat - future_deb[closest_debris_idx]
            escape_vector = escape_vector / np.linalg.norm(escape_vector)
            
            sat_vel = sat_v + (escape_vector * 0.02)
            break

    return warning

fig, ax = plt.subplots(figsize=(8, 8))
ax.set_xlim(-SPACE_SIZE, SPACE_SIZE)
ax.set_ylim(-SPACE_SIZE, SPACE_SIZE)
ax.set_title("Smart Orbit Management System - AI Collision Avoidance", fontsize=14, fontweight='bold')
ax.set_facecolor('black')

earth = plt.Circle((0, 0), 1.5, color='blue', alpha=0.5, label="Earth")
ax.add_patch(earth)

sat_plot, = ax.plot([], [], 'go', markersize=10, label="Satellite (Smart)")
deb_plot, = ax.plot([], [], 'ro', markersize=6, label="Space Debris")

status_text = ax.text(-9, 8.5, "", color='white', fontsize=12, fontweight='bold')
plt.legend(loc="upper right")

def init():
    sat_plot.set_data([], [])
    deb_plot.set_data([], [])
    status_text.set_text("")
    return sat_plot, deb_plot, status_text

def update(frame):
    global sat_pos, sat_vel, debris_pos, debris_vel
    
    debris_pos += debris_vel
    
    is_warning = predict_and_avoid(sat_pos, sat_vel, debris_pos, debris_vel)
    
    sat_pos += sat_vel
    
    sat_plot.set_data([sat_pos[0]], [sat_pos[1]])
    
    deb_x = debris_pos[:, 0]
    deb_y = debris_pos[:, 1]
    deb_plot.set_data(deb_x, deb_y)
    
    if is_warning:
        status_text.set_text("⚠️ WARNING: Collision Predicted! Maneuvering...")
        status_text.set_color('red')
        sat_plot.set_color('orange')
    else:
        status_text.set_text("✅ Orbit Clear. Nominal Flight.")
        status_text.set_color('green')
        sat_plot.set_color('green')
        
    return sat_plot, deb_plot, status_text

ani = animation.FuncAnimation(fig, update, frames=200, init_func=init, blit=True, interval=50)

plt.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.3)
plt.show()
