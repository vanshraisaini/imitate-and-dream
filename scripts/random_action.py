import numpy as np
import robosuite as suite

# Create the robot manipulation environment
env = suite.make(
    env_name="Lift",
    robots="Panda",
    has_renderer=True,
    has_offscreen_renderer=True,
    use_camera_obs=True,
    camera_names=["agentview", "robot0_eye_in_hand"],
    camera_heights=128,
    camera_widths=128,
    camera_depths=False,  # adds a *_depth entry per camera
    control_freq=20,
)

# Get action limits
low, high = env.action_spec

# Reset the environment
obs = env.reset()

print("Observation keys:", obs.keys())
print("Action dimension:", env.action_dim)
print(f"Action limits: {low} - {high}")

# Execute random actions
for step in range(1000):
    # Sample a random action within the valid action bounds
    action = np.random.uniform(low, high)

    # Advance the simulation
    obs, reward, done, info = env.step(action)

    # Render the simulation
    env.render()

    if step % 100 == 0:
        print(f"Step: {step}, Reward: {reward}")

    if done:
        obs = env.reset()

env.close()