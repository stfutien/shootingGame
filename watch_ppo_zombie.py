from stable_baselines3 import PPO
from rl_zombie_images_env import RLZombieImagesEnv
import time

def make_env():
    return RLZombieImagesEnv(width=160, height=120, render_scale=4,
                             player_hp=6, max_zombies=6, assets_path=".", use_sprites=True)

if __name__ == "__main__":
    env = make_env()

    # 🔹 Load trained model
    model = PPO.load("ppo_zombie/ppo_zombie_short")

    print("Watching trained agent play... (press Ctrl+C to stop)")

    # 🔹 Loop forever so you can watch continuously
    while True:
        obs = env.reset()
        done = False
        total_r = 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, r, done, _ = env.step(int(action))
            total_r += r
            env.render()
            time.sleep(0.02)  # slow down for visibility
        print("Episode reward:", total_r)
