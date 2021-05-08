import json
with open("configs/train.json", 'r')as f:
    config = json.load(f)

print(config.MOVES)