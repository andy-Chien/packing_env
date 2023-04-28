#!/usr/bin/env python3
import time
from stable_baselines3.common.env_checker import check_env
from packing_env import PackingEnv

def main():
    env = PackingEnv()
    # It will check your custom environment and output additional warnings if needed
    check_env(env)

if __name__ == '__main__':
    main()