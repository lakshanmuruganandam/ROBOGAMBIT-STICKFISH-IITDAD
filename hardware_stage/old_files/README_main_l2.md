# main_l2.py - Ultimate RoboGambit Hardware Controller

## Overview
This is the **ULTIMATE version** of the RoboGambit hardware controller with all fixes and improvements.

## Features
- ✅ **Fixed gripper crushing bug** - opens before lowering
- ✅ **Promotion tracking** - tracks promoted pieces
- ✅ **Precise pickup** - uses actual ArUco pose coordinates  
- ✅ **Board verification** - after every move
- ✅ **Performance metrics** - tracks success rate
- ✅ **Emergency recovery** - cleanup on errors
- ✅ **Homography logging** - reports all failures

## Usage
```bash
# Play as white
python main_l2.py --white

# Play as black
python main_l2.py --black

# Manual trigger mode
python main_l2.py --white --manual

# Calibration wizard
python main_l2.py --calibrate
```

## Files
- `main_l2.py` - Ultimate controller (USE THIS)
- `arm_controller_l2.py` - Arm controller with fixed gripper
- `perception_l2.py` - Perception with pose detection
- `config_l2.py` - Configuration (uses config_l1.py)

## Competition Rules Compliance
- 6×6 board with ArUco markers
- RoArm M2-S robotic arm
- Electromagnet gripper
- 15-minute game clock
- Setup phase for seeding
- Promotion: only to captured pieces
