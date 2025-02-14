from cedar.init import *
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog, messagebox


path = r"/Users/zhangsong/workspace/OpenSource/cedarpy/feature/可视化配方/config/config.json"
cfg = Config(path).config_data
print(cfg)
config_data = {}
for key, value in cfg.items():
    print("key", key, "value", value)
    cfg_value = Config(value).config_data
    config_data[key] = cfg_value

print(config_data)
