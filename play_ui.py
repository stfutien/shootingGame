# play_ui.py
import pygame
from rl_zombie_images_env import RLZombieImagesEnv

# Map keyboard state to the environment's move index (0..8)
# move_dirs order in env: 0 stop, then E, NE, N, NW, W, SW, S, SE


def get_move_index(keys):
    up = keys[pygame.K_w] or keys[pygame.K_UP]
    down = keys[pygame.K_s] or keys[pygame.K_DOWN]
    left = keys[pygame.K_a] or keys[pygame.K_LEFT]
    right = keys[pygame.K_d] or keys[pygame.K_RIGHT]

    # Diagonals first
    if right and up:
        return 2  # NE
    if left and up:
        return 4  # NW
    if left and down:
        return 6  # SW
    if right and down:
        return 8  # SE

    # Cardinals
    if right:
        return 1  # E
    if up:
        return 3  # N
    if left:
        return 5  # W
    if down:
        return 7  # S

    return 0  # no movement


def main():
    # You can tweak these to taste
    env = RLZombieImagesEnv(
        width=160, height=120, render_scale=4,
        # put player.png/zombie.png/wall.png here if you have them
        assets_path=".", use_sprites=True
    )

    # Make sure pygame is initialized before we start polling keys
    pygame.init()

    running = True
    shoot_cooldown = 0  # a tiny cooldown so holding space doesn't spam every single frame

    while running:
        # Poll events early so window responds instantly to close
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        keys = pygame.key.get_pressed()
        move_idx = get_move_index(keys)
        shoot = 0

        # Hold SPACE (or left mouse button) to shoot; limited by cooldown
        if (keys[pygame.K_SPACE] or pygame.mouse.get_pressed()[0]) and shoot_cooldown == 0:
            shoot = 1
            shoot_cooldown = 4  # ~4 frames at 60fps ≈ 67ms between shots

        action = move_idx * 2 + shoot

        # Step the env and render the UI
        _, _, done, _ = env.step(action)
        env.render()

        # Simple HUD key: press R to reset when dead or anytime
        if keys[pygame.K_r] or done:
            env.reset()

        # ESC to quit
        if keys[pygame.K_ESCAPE]:
            running = False

        if shoot_cooldown > 0:
            shoot_cooldown -= 1

    env.close()
    pygame.quit()


if __name__ == "__main__":
    main()
