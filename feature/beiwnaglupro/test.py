from pynput import keyboard
import pyautogui
import time
from cedar.supper import print

# 定义全局变量
command_pressed_time = None  # 记录 Command 键按下的时间
c_pressed_time = None  # 记录 C 键按下的时间


def on_press(key):
    global command_pressed_time, c_pressed_time

    try:
        # 检测 Command 键按下
        if key == keyboard.Key.cmd:
            command_pressed_time = time.time()  # 记录时间戳
        # 检测 C 键按下
        elif hasattr(key, "char") and key.char == "c":  # 检查是否有 char 属性
            c_pressed_time = time.time()  # 记录时间戳

        # 检测是否在1秒内同时按下 Command + C
        if command_pressed_time and c_pressed_time:
            if abs(command_pressed_time - c_pressed_time) <= 1:
                print("Detected Command + C within 1 second!")
                time.sleep(1.5)  # 等待1秒
                pyautogui.hotkey("command", "p")  # 执行 Command + 空格
                reset_keys()  # 重置按键状态
    except AttributeError:
        pass


def on_release(key):
    global command_pressed_time, c_pressed_time
    # 重置按键状态
    if key == keyboard.Key.cmd:
        command_pressed_time = None
    elif hasattr(key, "char") and key.char == "c":  # 检查是否有 char 属性
        c_pressed_time = None


def reset_keys():
    """重置按键状态"""
    global command_pressed_time, c_pressed_time
    command_pressed_time = None
    c_pressed_time = None


def main():
    with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
        listener.join()


if __name__ == "__main__":
    main()
