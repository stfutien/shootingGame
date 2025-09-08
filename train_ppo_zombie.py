# train_ppo_zombie.py
import os
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from rl_zombie_images_env import RLZombieImagesEnv

def make_env():
    return RLZombieImagesEnv(width=160, height=120, render_scale=4,
                             player_hp=6, max_zombies=6, assets_path=".", use_sprites=True)

if __name__ == "__main__":
    env = DummyVecEnv([lambda: make_env()])

    model = PPO("MlpPolicy", env, verbose=1, learning_rate=3e-4, batch_size=64, n_steps=2048)
    print("Starting training (short run)...")
    model.learn(total_timesteps=20000)   # increase to 200000+ for better results
    outdir = "ppo_zombie"
    os.makedirs(outdir, exist_ok=True)
    model.save(os.path.join(outdir, "ppo_zombie_short"))

    # 🔹 Run multiple demo episodes
    num_episodes = 5   # change this to watch more
    for ep in range(num_episodes):
        print(f"Demo Episode {ep+1}/{num_episodes}")
        demo_env = make_env()
        obs = demo_env.reset()
        done = False
        total_r = 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, r, done, _ = demo_env.step(int(action))
            total_r += r
            demo_env.render()
        print("Episode reward:", total_r)
        demo_env.close()

    print("All demo episodes finished ✅")
