# 安装依赖
# pip install pyobjc-framework-Quartz keyboard pyautogui

from Quartz import CGEventCreateKeyboardEvent, CGEventPost, kCGHIDEventTap
from AppKit import NSWorkspace
import keyboard
import time


def is_memo_active():
    return NSWorkspace.sharedWorkspace().frontmostApplication().localizedName() == "备忘录"


def send_keystroke(key, modifiers=[]):
    event = CGEventCreateKeyboardEvent(None, 0, True)
    CGEventPost(kCGHIDEventTap, event)
    time.sleep(0.05)


# 注册Ctrl+K代码块快捷键
keyboard.add_hotkey("ctrl+k", lambda: (keyboard.write("```\n\n```"), keyboard.press("up")) if is_memo_active() else None)


# 标题自动转换监听
def title_converter(e):
    if e.event_type == keyboard.KEY_DOWN and is_memo_active():
        current = keyboard.get_typed_strings(keyboard.record(until="space"))
        if list(current)[-1].endswith("# "):
            keyboard.press_and_release("backspace:2")  # 删除#和空格
            keyboard.press("command+t")  # 触发标题格式


keyboard.hook(title_converter)

# 保持脚本持续运行
keyboard.wait()
