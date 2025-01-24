import json
import pandas as pd
import matplotlib.pyplot as plt

# Step 1: Read the JSONL file
data = []
with open("/storage/abdulw/SkyThought/skythought/train/LLaMA-Factory/outputs/phi3/full/original/trainer_log.jsonl", "r") as file:
    for line in file:
        data.append(json.loads(line))

# Step 2: Extract `current_steps` and `loss`
steps = [entry["current_steps"] for entry in data]
loss = [entry["loss"] for entry in data]

# Step 3: Store in a pandas DataFrame
df = pd.DataFrame({"current_steps": steps, "loss": loss})

# Step 4: Calculate moving average (e.g., window size = 10)
window_size = 100  # Adjust this value to control the smoothness
df["loss_smooth"] = df["loss"].rolling(window=window_size, min_periods=1).mean()

# Step 5: Plot the data
plt.figure(figsize=(10, 6))
# plt.plot(df["current_steps"], df["loss"], marker="o", linestyle="-", color="b", alpha=0.3, label="Original Loss")
plt.plot(df["current_steps"], df["loss_smooth"], linestyle="-", color="r", linewidth=2, label=f"Smoothed Loss (Window={window_size})")
plt.title("Phi-3.5-mini-instruct + O1")
plt.xlabel("Current Steps")
plt.ylabel("Loss")
plt.grid(True)
plt.legend()
plt.savefig("plot_smoothed.png")