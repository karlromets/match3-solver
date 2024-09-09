from pynput.mouse import Listener, Button
import sys

# Function called on a mouse click
def on_click(x, y, button, pressed):
    # Check if the left button was pressed
    if pressed and button == Button.left:
        # Print the click coordinates
        print(f'x={x} and y={y}')
    # Check if the right button was pressed to exit the program
    if pressed and button == Button.right:
        print("Exiting program.")
        listener.stop()
        sys.exit()

# Initialize the Listener to monitor mouse clicks
with Listener(on_click=on_click) as listener:
    listener.join()