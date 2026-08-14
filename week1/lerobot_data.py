from lerobot.datasets.lerobot_dataset import LeRobotDataset

dataset = LeRobotDataset(repo_id="lerobot/pusht")

print(dataset)
print(dataset.features)

sample = dataset[0]
print(sample.keys())

for key, value in sample.items():
    print(
        key,
        "shape =", getattr(value, "shape", None),
        "dtype =", getattr(value, "dtype", None),
    )

# 查看任务文字或任务索引
print("tasks:", getattr(dataset.meta, "tasks", None))

# 查看实际缓存位置
print("local root:", getattr(dataset, "root", None))
