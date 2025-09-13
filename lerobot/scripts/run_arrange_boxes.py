import gymnasium as gym
import gym_hil
from lerobot.policies.pretrained import PreTrainedPolicy

from lerobot.policies.pretrained import PreTrainedPolicy

def main():
    env = gym.make("gym_hil/PandaArrangeBoxesKeyboard-v0", render_mode="human")
    obs, _ = env.reset(seed=42)
    print("Observaciones:", list(obs.keys()))

    policy = PreTrainedPolicy.from_pretrained("lerobot/smolvla_base")
    policy.reset()

    done = False
    while not done:
        action = policy.select_action(obs)  # Ajustar según keys de tu obs
        obs, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        env.render()
    env.close()

if __name__ == "__main__":
    main()
