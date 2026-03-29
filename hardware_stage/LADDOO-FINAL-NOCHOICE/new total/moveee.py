import pygame
import json
import time
import math
import sys

# Import YOUR existing variables and functions
from main import ser, ser2, get_feedback_full, send_cmd, EOAT_LEVEL, electromagnet_off, electromagnet_on

# --- CONFIG ---
STEP = 7.0       # mm to move per tick while holding key
T_STEP = 0.05    # radians to tilt per tick
FPS = 30       # Speed of the loop
 
# Initialize Pygame
pygame.init()
screen = pygame.display.set_mode((400, 300))
pygame.display.set_caption("Robot Jogger + Magnet")
clock = pygame.time.Clock()

def jog_move(x, y, z, t):
    """Uses your existing send_cmd logic."""
    cmd = f'{{"T":1041,"x":{x:.3f},"y":{y:.3f},"z":{z:.3f},"t":{t:.5f}}}'
    send_cmd(cmd)

def set_magnet(state: bool):
    """Sends 1 or 0 to ser2 for the solenoid."""
    # print(f"Setting Magnet {'ON' if state else 'OFF'}...")
    if state:
        electromagnet_on()
    else: 
        electromagnet_off()

print("\n--- JOGGER READY ---")
print("W / S : Move X")
print("A / D : Move Y")
print("R / F : Move Z (Height)")
print("Q / E : Tilt Wrist")
print("SPACE : TOGGLE MAGNET")
print("P     : PRINT COORDINATES")
print("ESC   : Exit")

# Initial Position Sync
ax, ay, az, as_angle, ae_angle = get_feedback_full()
if ax is None:
    ax, ay, az = 300.0, 0.0, 150.0

current_t = (math.pi/2) - as_angle + ae_angle - EOAT_LEVEL
magnet_on = False

running = True
while running:
    moved = False
    keys = pygame.key.get_pressed()

    # eot_angle_corrected = math.pi


    # --- Continuous Movement Logic ---
    if keys[pygame.K_w]: ax += STEP; moved = True
    if keys[pygame.K_s]: ax -= STEP; moved = True
    if keys[pygame.K_a]: ay += STEP; moved = True
    if keys[pygame.K_d]: ay -= STEP; moved = True
    if keys[pygame.K_r]: az += STEP; moved = True
    if keys[pygame.K_f]: az -= STEP; moved = True
    if keys[pygame.K_q]: current_t += T_STEP; moved = True
    if keys[pygame.K_e]: current_t -= T_STEP; moved = True
    # --- Event Handling (Single Press) ---
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN:
            # MAGNET TOGGLE
            if event.key == pygame.K_SPACE:
                magnet_on = not magnet_on
                set_magnet(magnet_on)
                
            if event.key == pygame.K_p:
                fx, fy, fz, fs, fe = get_feedback_full()
                # eot_angle_corrected = (math.pi/2) - fs + fe
                print(f"\n[CALIBRATION FEEDBACK]")
                print(f"X: {fx:.2f}, Y: {fy:.2f}, Z: {fz:.2f}")
                print(f"Current Level 't': {current_t:.5f}")
                
            if event.key == pygame.K_ESCAPE:
                running = False

    if moved:
        jog_move(ax, ay, az, current_t)
        time.sleep(0.08) 

    pygame.display.update()
    clock.tick(FPS)

# Safety: Turn off magnet on exit
set_magnet(False)
pygame.quit()