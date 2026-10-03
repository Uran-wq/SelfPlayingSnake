import pygame
import sys
import time
from environment import SnakeEnv, Action


CELL_SIZE = 24
INFO_HEIGHT = 80
FPS_INITIAL = 10

COLOR_BG = (30, 30, 30)
COLOR_GRID = (40, 40, 40)
COLOR_HEAD = (0, 220, 0)
COLOR_BODY = (0, 160, 0)
COLOR_FOOD = (220, 30, 30)
COLOR_TEXT = (220, 220, 220)
COLOR_GAMEOVER = (220, 50, 50)
COLOR_SCORE = (255, 215, 0)


class GameVisualizer:
    def __init__(self, env: SnakeEnv, agent, title: str = "Snake", fps: int = FPS_INITIAL, neural: bool = False):
        self.env = env
        self.agent = agent
        self.title = title
        self.fps = fps
        self.grid_size = env.grid_size
        self.neural = neural

        self.cell_size = CELL_SIZE
        self.width = self.grid_size * self.cell_size
        self.height = self.grid_size * self.cell_size + INFO_HEIGHT

        pygame.init()
        pygame.display.set_caption(title)
        self.screen = pygame.display.set_mode((self.width, self.height))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("consolas", 16, bold=True)
        self.font_big = pygame.font.SysFont("consolas", 22, bold=True)

        self.paused = False
        self.speed = fps
        self.running = True

    def run_episode(self) -> dict:
        self.env.reset()
        self.agent.reset() if hasattr(self.agent, 'reset') else None

        total_reward = 0
        steps = 0
        done = False

        while self.running and not done:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                    pygame.quit()
                    return {"score": self.env.score, "steps": steps, "done_early": True}
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        self.paused = not self.paused
                    elif event.key == pygame.K_UP:
                        self.speed = min(self.speed + 5, 120)
                    elif event.key == pygame.K_DOWN:
                        self.speed = max(self.speed - 5, 1)
                    elif event.key == pygame.K_ESCAPE:
                        self.running = False
                        pygame.quit()
                        return {"score": self.env.score, "steps": steps, "done_early": True}

            if self.paused:
                self.clock.tick(10)
                continue

            if self.neural:
                state = self.env.get_simplified_observation()
                action = self.agent.act(state, training=False)
            else:
                action = self.agent.act(self.env)

            _, reward, done, info = self.env.step(action)
            total_reward += reward
            steps += 1

            self._draw(steps)

            if self.env.score > 0:
                self.clock.tick(self.speed)
            else:
                self.clock.tick(min(self.speed, 30))

        if self.running:
            self._draw_final(steps)
            self._wait_for_close()

        return {
            "score": info.get("score", self.env.score),
            "steps": steps,
            "total_reward": total_reward,
            "done_early": False,
        }

    def run_n_episodes(self, n: int = 5) -> list:
        results = []
        for i in range(n):
            if not self.running:
                break
            result = self.run_episode()
            result["episode"] = i + 1
            results.append(result)
            if not self.running:
                break
            self._show_episode_summary(result, i + 1, n)
        return results
    
    def _draw(self, steps: int) -> None:
        self.screen.fill(COLOR_BG)

        for r in range(self.grid_size):
            for c in range(self.grid_size):
                rect = pygame.Rect(c * self.cell_size, r * self.cell_size, self.cell_size, self.cell_size)
                pygame.draw.rect(self.screen, COLOR_GRID, rect)
                pygame.draw.rect(self.screen, (50, 50, 50), rect, 1)

        food_r, food_c = self.env.food
        food_rect = pygame.Rect(food_c * self.cell_size + 2, food_r * self.cell_size + 2,
                                 self.cell_size - 4, self.cell_size - 4)
        pygame.draw.rect(self.screen, COLOR_FOOD, food_rect, border_radius=5)

        for i, (r, c) in enumerate(self.env.snake):
            rect = pygame.Rect(c * self.cell_size + 1, r * self.cell_size + 1,
                               self.cell_size - 2, self.cell_size - 2)
            color = COLOR_HEAD if i == 0 else COLOR_BODY
            radius = 4 if i == 0 else 2
            pygame.draw.rect(self.screen, color, rect, border_radius=radius)

        info_y = self.grid_size * self.cell_size
        pygame.draw.rect(self.screen, (20, 20, 20), (0, info_y, self.width, INFO_HEIGHT))

        score_text = self.font_big.render(f"Score: {self.env.score}", True, COLOR_SCORE)
        self.screen.blit(score_text, (10, info_y + 5))

        steps_text = self.font.render(f"Steps: {steps}", True, COLOR_TEXT)
        self.screen.blit(steps_text, (10, info_y + 35))

        speed_text = self.font.render(f"Speed: {self.speed} [UP/DOWN]", True, COLOR_TEXT)
        self.screen.blit(speed_text, (180, info_y + 35))

        pause_text = self.font.render("[SPACE] Pause  [ESC] Quit", True, (150, 150, 150))
        self.screen.blit(pause_text, (180, info_y + 55))

        title_text = self.font.render(self.title, True, (100, 180, 100))
        self.screen.blit(title_text, (180, info_y + 5))

        if self.paused:
            pause_label = self.font_big.render("PAUSED", True, (255, 255, 0))
            pw = pause_label.get_width()
            self.screen.blit(pause_label, (self.width // 2 - pw // 2, info_y + 8))

        pygame.display.flip()

    def _draw_final(self, steps: int) -> None:
        self._draw(steps)
        overlay = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 150))
        self.screen.blit(overlay, (0, 0))

        go_text = self.font_big.render("GAME OVER", True, COLOR_GAMEOVER)
        self.screen.blit(go_text, (self.width // 2 - go_text.get_width() // 2, self.height // 2 - 40))

        final_score = self.font_big.render(f"Score: {self.env.score}", True, COLOR_SCORE)
        self.screen.blit(final_score, (self.width // 2 - final_score.get_width() // 2, self.height // 2))

        hint = self.font.render("Press any key to continue", True, COLOR_TEXT)
        self.screen.blit(hint, (self.width // 2 - hint.get_width() // 2, self.height // 2 + 40))

        pygame.display.flip()

    def _show_episode_summary(self, result: dict, current: int, total: int) -> None:
        self.screen.fill(COLOR_BG)
        info_y = self.grid_size * self.cell_size
        pygame.draw.rect(self.screen, (20, 20, 20), (0, 0, self.width, self.height))

        title = self.font_big.render(f"Episode {current}/{total}", True, COLOR_TEXT)
        self.screen.blit(title, (self.width // 2 - title.get_width() // 2, 30))

        score = self.font_big.render(f"Score: {result['score']}", True, COLOR_SCORE)
        self.screen.blit(score, (self.width // 2 - score.get_width() // 2, 70))

        length = self.font.render(f"Steps: {result['steps']}", True, COLOR_TEXT)
        self.screen.blit(length, (self.width // 2 - length.get_width() // 2, 110))

        hint = self.font.render("Press SPACE to run next episode", True, (150, 150, 150))
        self.screen.blit(hint, (self.width // 2 - hint.get_width() // 2, 160))

        pygame.display.flip()

        waiting = True
        while waiting and self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                    waiting = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        self.running = False
                        waiting = False
                    else:
                        waiting = False
            self.clock.tick(15)

    def _wait_for_close(self) -> None:
        waiting = True
        while waiting and self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                    waiting = False
                elif event.type == pygame.KEYDOWN:
                    waiting = False
            self.clock.tick(15)

    def close(self):
        self.running = False
        pygame.quit()


def visualize_episode(env: SnakeEnv, agent, title: str = "Snake", fps: int = 10, n_episodes: int = 1, neural: bool = False) -> list:
    viz = GameVisualizer(env, agent, title=title, fps=fps, neural=neural)
    results = viz.run_n_episodes(n_episodes)
    viz.close()
    return results