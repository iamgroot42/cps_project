# Security in AI-aided Autoland

Course Project for Cyber-Physical Systems (CPS) Spring 2021 @ UVa


## Instructions

1. Run `python create_adv_example.py N` for some number N to generate a log for adversary success with distance. Run this multiple times for varying N to get multiple readings.
2. Edit `config.json` to edit details like width of landing strip, maximum angle restrictions, distance multiplier (to sync warp-distance from images and actual data)
3. Run `estimate_adv_prob.py` to generate `.pm` file that can then be used in PRISM
