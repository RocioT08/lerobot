import gymnasium as gym
from huggingface_hub import hf_hub_download
from lerobot.policies.pretrained import PreTrainedPolicy

def main():
    # 1. Crea tu entorno de simulación
    env = gym.make("PandaArrangeBoxesKeyboard-v0", render_mode="human")
    obs, _ = env.reset(seed=42)
    print("🔎 Observaciones disponibles:", list(obs.keys()))

    # 2. Descarga y carga el modelo desde Hugging Face
    repo_id = "aiden-li/so101-act"
    ckpt_file = hf_hub_download(repo_id=repo_id, filename="model.safetensors")

    policy = PreTrainedPolicy.from_pretrained(repo_id, device="cpu")
    print(f"✅ Modelo {repo_id} cargado con éxito")

    # 3. Bucle de simulación
    done = False
    while not done:
        action = policy.predict(obs)  # La política genera la acción
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        env.render()

    env.close()

if __name__ == "__main__":
    main()
