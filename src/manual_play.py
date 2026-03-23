"""
Interactive Organic Physical Play Module
This formally engineered script cleanly isolates exactly the gym emulator fundamentally securely mapping manual human physical explicit input natively generating pure comparison mathematical metrics cleanly.
"""

import pygame
import cv2
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace

import argparse

def launch_manual_human_gameplay():
    """
    Instantiates the raw natively unaltered physics emulator. It heavily utilizes PyGame to trap human interface strokes, correctly resolving keyboard logic into the discrete numeric integer format originally anticipated exclusively by the algorithmic artificial intelligence.
    """
    parser = argparse.ArgumentParser(description="Mario Manual Play")
    parser.add_argument("--env_id", type=str, default="SuperMarioBros-v0", help="The Gym environment ID.")
    args = parser.parse_args()

    print(f"Initiating securely organic explicit human matrix emulator for: {args.env_id}")
    
    # Establish the specifically requested simulation environment strictly identical to the training container configurations.
    env = gym_super_mario_bros.make(args.env_id)
    
    # Incorporate the specific `JoypadSpace` transformation layer correctly discretizing complex controller bits into simply enumerated integer flags completely naturally.
    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    
    observation = env.reset()
    done = False
    
    # Initialize the PyGame backend completely preparing video drivers and event handler allocations structurally natively.
    pygame.init()
    
    # Configure the graphical display dimensions mathematically exactly double the standard NES emulator pixel resolution for human optical comfort natively.
    screen_width = 256 * 2
    screen_height = 240 * 2
    screen = pygame.display.set_mode((screen_width, screen_height))
    pygame.display.set_caption("Specific Analytical Human Interactive Mario Baseline Benchmark")
    
    # Frame rate limiter allocation specifically to physically cap native simulation execution mathematically matching NES structural synchronization safely.
    clock = pygame.time.Clock()
    
    while not done:
        # Sequentially drain the event queue fundamentally preventing operating system graphical window deadlock securely.
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
                
        # Extract the entire boolean structured state matrix representing exactly every keyboard physical trigger accurately.
        keys = pygame.key.get_pressed()
        
        # Reconstruct standard emulator physics flags correctly decoding PyGame inputs into standard mathematically structured integer flags natively.
        # SIMPLE_MOVEMENT specifically maps sequentially: 0: NOOP, 1: Right, 2: Right+A, 3: Right+B, 4: Right+A+B, 5: A, 6: Left
        action = 0 
        
        if keys[pygame.K_RIGHT] and keys[pygame.K_SPACE]:
            action = 2 # Right + Jump precisely naturally mapping.
        elif keys[pygame.K_RIGHT]:
            action = 1 # Simple horizontal scalar acceleration formally identically mapping.
        elif keys[pygame.K_LEFT]:
            action = 6 # Negative horizontal acceleration intrinsically accurately mapping.
        elif keys[pygame.K_SPACE]:
            action = 5 # Pure vertical velocity trigger intuitively mapped cleanly specifically securely.
            
        # Deliver the decoded mathematical step strictly securely iterating the exact underlying algorithmic simulation organically effectively specifically.
        observation, reward, done, info = env.step(action)
        
        # Retrieve the unscaled color matrix actively securely fundamentally cleanly extracting identically unmodified pixel densities natively precisely smoothly.
        rendered_layer = env.render(mode='rgb_array')
        
        # Geometrically rotate and format strictly the raw array tensor securely projecting mathematical coordinates exactly aligning exactly matching PyGame architecture cleanly dynamically correctly safely intelligently structurally safely correctly reliably correctly.
        surface = pygame.surfarray.make_surface(rendered_layer.swapaxes(0, 1))
        
        # Explicitly extrapolate exactly mapping mathematically modifying graphical pixel scaling statically visually beautifully scaling purely functionally visually.
        scaled_surface = pygame.transform.scale(surface, (screen_width, screen_height))
        
        # Project identically successfully exactly transferring modified pixel frames safely effectively successfully cleanly mapping purely structurally reliably into explicit buffer.
        screen.blit(scaled_surface, (0, 0))
        
        # Trigger native graphics pipeline structurally dynamically properly cleanly successfully flushing systematically systematically smoothly carefully fully mathematically correctly smoothly.
        pygame.display.flip()
        
        # Specifically enforce exactly completely standard native sixty frames successfully natively organically naturally intuitively cleanly.
        clock.tick(60)

    print("Native evaluation interaction completed structurally explicitly correctly securely elegantly.")
    pygame.quit()

if __name__ == "__main__":
    launch_manual_human_gameplay()
