"""
Manual Play Module
Allows a human player to control Mario using keyboard input via PyGame.
Resets automatically on death so the player can keep playing.
"""

import pygame
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace
import argparse


def launch_manual_gameplay():
    """Opens a PyGame window for direct human control of Mario."""
    parser = argparse.ArgumentParser(description="Mario Manual Play")
    parser.add_argument("--env_id", type=str, default="SuperMarioBros-v0",
                        help="The Gym environment ID.")
    args = parser.parse_args()

    print(f"Starting manual play: {args.env_id}")
    print("Controls: Arrow keys to move, Space/Up to jump, Shift to sprint.")
    print("Close the window or press Escape to quit.")

    env = gym_super_mario_bros.make(args.env_id)
    env = JoypadSpace(env, SIMPLE_MOVEMENT)

    observation = env.reset()

    pygame.init()
    screen_width = 256 * 2
    screen_height = 240 * 2
    screen = pygame.display.set_mode((screen_width, screen_height))
    pygame.display.set_caption("Mario - Manual Play")
    clock = pygame.time.Clock()

    # SIMPLE_MOVEMENT action mapping:
    #   0: NOOP
    #   1: Right
    #   2: Right + A (jump)
    #   3: Right + B (sprint)
    #   4: Right + A + B (sprint jump)
    #   5: A (jump in place)
    #   6: Left

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False

        if not running:
            break

        keys = pygame.key.get_pressed()

        # Determine action based on key combinations
        right = keys[pygame.K_RIGHT]
        left = keys[pygame.K_LEFT]
        jump = keys[pygame.K_SPACE] or keys[pygame.K_UP]
        sprint = keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]

        if right and jump and sprint:
            action = 4  # Right + Jump + Sprint
        elif right and jump:
            action = 2  # Right + Jump
        elif right and sprint:
            action = 3  # Right + Sprint
        elif right:
            action = 1  # Right
        elif left:
            action = 6  # Left
        elif jump:
            action = 5  # Jump in place
        else:
            action = 0  # NOOP (stand still)

        observation, reward, done, info = env.step(action)

        # Reset on death instead of ending the session
        if done:
            observation = env.reset()

        rendered_frame = env.render(mode='rgb_array')
        surface = pygame.surfarray.make_surface(rendered_frame.swapaxes(0, 1))
        scaled_surface = pygame.transform.scale(surface, (screen_width, screen_height))
        screen.blit(scaled_surface, (0, 0))
        pygame.display.flip()
        clock.tick(60)

    print("Manual play session ended.")
    pygame.quit()
    env.close()


if __name__ == "__main__":
    launch_manual_gameplay()
