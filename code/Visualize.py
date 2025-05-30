import json
import matplotlib.pyplot as plt

# 加载 trainer_state.json 文件
state_file = "./data/checkpoint-52818/trainer_state.json"
with open(state_file, "r") as f:
    state = json.load(f)

# 提取日志信息
log_history = state.get("log_history", [])
loss_points = [(log["step"], log["loss"]) for log in log_history if "loss" in log]

# 拆分 step 和 loss 为两个数组
steps, losses = zip(*loss_points)

# 绘图
plt.figure(figsize=(8, 5))
plt.plot(steps, losses, marker="o")
plt.xlabel("Training Step")
plt.ylabel("Loss")
plt.title("Training Loss Curve")
plt.grid(True)
plt.tight_layout()
plt.show()
plt.savefig("loss_curve.png")
