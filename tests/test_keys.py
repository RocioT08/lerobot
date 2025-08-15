from pynput import keyboard
def on_press(key): print("PRESS:", key)
def on_release(key): print("RELEASE:", key)
with keyboard.Listener(on_press=on_press, on_release=on_release) as L:
    L.join()

