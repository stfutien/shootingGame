# rl_zombie_images_env.py
import os
import math
import random
import numpy as np
import pygame
import gym
from gym import spaces

class RLZombieImagesEnv(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(self,
                 width=160, height=120, render_scale=4,
                 player_speed=2.0, bullet_speed=6.0, zombie_speed=0.9,
                 player_hp=6, max_zombies=6, max_steps=600,
                 walls=None,
                 assets_path=".", use_sprites=True):
        super().__init__()
        self.W = width
        self.H = height
        self.render_scale = render_scale
        self.player_speed = player_speed
        self.bullet_speed = bullet_speed
        self.zombie_speed = zombie_speed
        self.start_hp = player_hp
        self.max_zombies = max_zombies
        self.max_steps = max_steps
        self.assets_path = assets_path
        self.use_sprites = use_sprites

        # physics radii
        self.player_r = 8
        self.zombie_r = 8
        self.bullet_r = 2

        # actions: (move_idx 0..8) * 2 (shoot/no-shoot)
        self.move_dirs = [(0,0)] + [(math.cos(a), math.sin(a)) for a in [i*math.pi/4 for i in range(8)]]
        self.aim_dirs = [(math.cos(a), math.sin(a)) for a in [i*math.pi/4 for i in range(8)]]
        self.n_move = len(self.move_dirs)
        self.n_actions = self.n_move * 2
        self.action_space = spaces.Discrete(self.n_actions)

        # observation: same compact vector as before
        # obs vector length: 4 base + K*3
        self.K = 4
        obs_len = 4 + self.K * 3
        high = np.ones(obs_len, dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_len,), dtype=np.float32)

        # walls: list of pygame.Rect-like tuples (x,y,w,h), default sample walls
        if walls is None:
    
            walls = [
                (40, 30, 20, 20),   # small wall block
                (90, 60, 20, 20),   # another small wall block
                (120, 80, 15, 15),  # tiny obstacle
            ]
        self.walls = [pygame.Rect(*w) for w in walls]

        # rendering fields
        self.screen = None
        self.clock = None

        # sprites (load lazily)
        self.player_img = None
        self.zombie_img = None
        self.wall_img = None

        self.reset()

    def _load_sprites(self):
        if not self.use_sprites:
            return
        try:
            # attempt to load images; fall back to circle draws if missing
            ppath = os.path.join(self.assets_path, "player.png")
            zpath = os.path.join(self.assets_path, "zombie.png")
            wpath = os.path.join(self.assets_path, "wall.png")
            if os.path.exists(ppath):
                img = pygame.image.load(ppath).convert_alpha()
                self.player_img = pygame.transform.scale(img, (self.player_r*2, self.player_r*2))
            if os.path.exists(zpath):
                img = pygame.image.load(zpath).convert_alpha()
                self.zombie_img = pygame.transform.scale(img, (self.zombie_r*2, self.zombie_r*2))
            if os.path.exists(wpath):
                img = pygame.image.load(wpath).convert()
                # tile to wall rect size at draw time
                self.wall_img = img
        except Exception as e:
            print("Warning: sprite load failed:", e)
            self.player_img = None
            self.zombie_img = None
            self.wall_img = None

    def reset(self):
        self.player_x = self.W/2.0
        self.player_y = self.H/2.0
        self.player_hp = self.start_hp
        self.bullets = []   # dicts: x,y,vx,vy,life
        self.zombies = []   # dicts: x,y,health
        for _ in range(3):
            self._spawn_zombie()
        self.steps = 0
        self.done = False
        self.total_reward = 0.0

        # lazy sprite load
        if self.player_img is None and self.use_sprites:
            pygame.init()
            self._load_sprites()

        return self._get_obs()

    def _spawn_zombie(self):
        side = random.choice(['left','right','top','bottom'])
        if side == 'left':
            x = -10; y = random.uniform(0, self.H)
        elif side == 'right':
            x = self.W + 10; y = random.uniform(0, self.H)
        elif side == 'top':
            x = random.uniform(0, self.W); y = -10
        else:
            x = random.uniform(0, self.W); y = self.H + 10
        self.zombies.append({'x':x, 'y':y, 'health':1})

    def _get_obs(self):
        px, py = self.player_x, self.player_y
        zsorted = sorted(self.zombies, key=lambda z: (z['x']-px)**2 + (z['y']-py)**2)
        obs = [px/self.W, py/self.H, self.player_hp / float(self.start_hp), len(self.bullets)/10.0]
        for i in range(self.K):
            if i < len(zsorted):
                zx, zy = zsorted[i]['x'], zsorted[i]['y']
                dx = (zx - px) / self.W
                dy = (zy - py) / self.H
                dist = math.hypot(zx-px, zy-py) / math.hypot(self.W, self.H)
                obs += [dx, dy, dist]
            else:
                obs += [0.0, 0.0, 1.0]
        return np.array(obs, dtype=np.float32)

    def step(self, action):
        if self.done:
            return self._get_obs(), 0.0, True, {}

        move_idx = int(action) // 2
        shoot_flag = int(action) % 2

        # move player
        mv = self.move_dirs[move_idx]
        new_x = self.player_x + mv[0] * self.player_speed
        new_y = self.player_y + mv[1] * self.player_speed

        # clamp to screen boundaries
        new_x = max(self.player_r, min(self.W - self.player_r, new_x))
        new_y = max(self.player_r, min(self.H - self.player_r, new_y))

        # collision with walls: try move then revert if collides
        player_rect = pygame.Rect(new_x - self.player_r, new_y - self.player_r, self.player_r*2, self.player_r*2)
        collided = False
        for w in self.walls:
            if player_rect.colliderect(w):
                collided = True
                break
        if not collided:
            self.player_x, self.player_y = new_x, new_y


        # shoot
        if shoot_flag:
            if len(self.zombies) > 0:
                zx = self.zombies[0]['x']; zy = self.zombies[0]['y']
                ang = math.atan2(zy - self.player_y, zx - self.player_x)
                best = min(range(len(self.aim_dirs)),
                           key=lambda i: abs(math.atan2(self.aim_dirs[i][1], self.aim_dirs[i][0]) - ang))
                dx, dy = self.aim_dirs[best]
            else:
                dx, dy = random.choice(self.aim_dirs)
            bx = self.player_x + dx*(self.player_r+2)
            by = self.player_y + dy*(self.player_r+2)
            self.bullets.append({'x':bx, 'y':by, 'vx':dx*self.bullet_speed, 'vy':dy*self.bullet_speed, 'life':40})

        # move bullets & bullet-wall collisions
        for b in self.bullets[:]:
            b['x'] += b['vx']; b['y'] += b['vy']; b['life'] -= 1
            # check wall collisions
            br = pygame.Rect(int(b['x']-self.bullet_r), int(b['y']-self.bullet_r), self.bullet_r*2, self.bullet_r*2)
            hit_wall = False
            for w in self.walls:
                if br.colliderect(w):
                    hit_wall = True
                    break
            if hit_wall or b['life'] <= 0 or b['x'] < -20 or b['x'] > self.W+20 or b['y'] < -20 or b['y'] > self.H+20:
                try:
                    self.bullets.remove(b)
                except ValueError:
                    pass

        # move zombies with wall avoidance (simple: revert move if collides)
        for z in self.zombies:
            dx = self.player_x - z['x']; dy = self.player_y - z['y']
            dist = math.hypot(dx, dy) + 1e-6
            nx = z['x'] + (dx/dist) * self.zombie_speed
            ny = z['y'] + (dy/dist) * self.zombie_speed
            zrect = pygame.Rect(nx-self.zombie_r, ny-self.zombie_r, self.zombie_r*2, self.zombie_r*2)
            hits = False
            for w in self.walls:
                if zrect.colliderect(w):
                    hits = True
                    break
            if not hits:
                z['x'], z['y'] = nx, ny
            # else do not move this step

        # bullet -> zombie collisions
        reward = 0.0
        for b in self.bullets[:]:
            for z in self.zombies[:]:
                if (b['x']-z['x'])**2 + (b['y']-z['y'])**2 <= (self.bullet_r + self.zombie_r)**2:
                    if b in self.bullets:
                        self.bullets.remove(b)
                    if z in self.zombies:
                        self.zombies.remove(z)
                    reward += 1.0
                    break

        # zombie -> player collisions (bites)
        for z in self.zombies[:]:
            if (z['x']-self.player_x)**2 + (z['y']-self.player_y)**2 <= (self.zombie_r + self.player_r)**2:
                self.player_hp -= 1
                ang = math.atan2(z['y'] - self.player_y, z['x'] - self.player_x)
                z['x'] += math.cos(ang) * 8
                z['y'] += math.sin(ang) * 8
                reward -= 0.5

        # spawn occasionally
        if random.random() < 0.02 and len(self.zombies) < self.max_zombies:
            self._spawn_zombie()

        self.steps += 1
        if self.player_hp <= 0:
            self.done = True
            reward -= 5.0
        elif self.steps >= self.max_steps:
            self.done = True

        self.total_reward += reward
        return self._get_obs(), reward, self.done, {}

    def render(self, mode="human"):
    # 🔹 Handle window events so game doesn't freeze
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()
        if self.screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode((self.W*self.render_scale, self.H*self.render_scale))
            pygame.display.set_caption("RL Zombie Images")
            self.clock = pygame.time.Clock()
            # ensure sprites loaded
            if self.use_sprites and self.player_img is None:
                self._load_sprites()

        surf = pygame.Surface((self.W, self.H))
        surf.fill((18, 18, 28))

        # draw walls (textured if wall_img present else gray rects)
        for w in self.walls:
            if self.wall_img is not None:
                # simple fill: scale texture to rect size (could tile)
                tex = pygame.transform.scale(self.wall_img, (w.width, w.height))
                surf.blit(tex, (w.x, w.y))
            else:
                pygame.draw.rect(surf, (170,170,170), w)

        # draw bullets
        for b in self.bullets:
            pygame.draw.circle(surf, (255, 230, 100), (int(b['x']), int(b['y'])), self.bullet_r)

        # draw zombies with sprite or circle
        for z in self.zombies:
            if self.zombie_img is not None:
                rect = self.zombie_img.get_rect(center=(int(z['x']), int(z['y'])))
                surf.blit(self.zombie_img, rect.topleft)
            else:
                pygame.draw.circle(surf, (200, 60, 60), (int(z['x']), int(z['y'])), self.zombie_r)

        # draw player with sprite or circle
        if self.player_img is not None:
            rect = self.player_img.get_rect(center=(int(self.player_x), int(self.player_y)))
            surf.blit(self.player_img, rect.topleft)
        else:
            pygame.draw.circle(surf, (60, 200, 60), (int(self.player_x), int(self.player_y)), self.player_r)

        # HUD
        font = pygame.font.SysFont(None, 18)
        txt = font.render(f"HP: {self.player_hp}  Steps: {self.steps}  Zombies: {len(self.zombies)}", True, (220,220,220))
        surf.blit(txt, (4,4))

        # scale to render_scale and blit
        surf = pygame.transform.scale(surf, (self.W*self.render_scale, self.H*self.render_scale))
        self.screen.blit(surf, (0,0))
        pygame.display.flip()
        self.clock.tick(60)

    def close(self):
        if self.screen:
            pygame.quit()
            self.screen = None
