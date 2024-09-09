import pyautogui
import cv2
import numpy as np
import time
from pynput import keyboard

class Match3Solver:
    def __init__(self):
        self.templates = self.load_templates()
        self.board_region = self.detect_game_board()
        self.colors = {
            'object1': (255, 0, 0), # Blue
            'object2': (0, 255, 0), # Green
            'object3': (0, 0, 255), # Red
            'object4': (255, 255, 0), # Cyan
            'object5': (0, 255, 255), # Yellow
            'object6': (255, 0, 255)  # Magenta
        }
        self.running = True
        self.setup_key_listener()

    def load_templates(self):
        templates = {
            'object1': cv2.imread('object1_template.png'),
            'object2': cv2.imread('object2_template.png'),
            'object3': cv2.imread('object3_template.png'),
            'object4': cv2.imread('object4_template.png'),
            'object5': cv2.imread('object5_template.png'),
            'object6': cv2.imread('object6_template.png')
        }
        for name, template in templates.items():
            if template is None:
                print(f"Failed to load template: {name}")
            else:
                print(f"Loaded template: {name} - Shape: {template.shape}")
        return templates

    def detect_game_board(self):
        screen = np.array(pyautogui.screenshot())
        screen_rgb = cv2.cvtColor(screen, cv2.COLOR_BGR2RGB)
        template = cv2.imread('board_template.png')
        template_rgb = cv2.cvtColor(template, cv2.COLOR_BGR2RGB)
        res = cv2.matchTemplate(screen_rgb, template_rgb, cv2.TM_CCOEFF_NORMED)
        _, _, _, max_loc = cv2.minMaxLoc(res)
        top_left = max_loc
        h, w = template_rgb.shape[:2]
        return (top_left[0], top_left[1], w, h)

    def locate_objects(self):
        board_img = np.array(pyautogui.screenshot(region=self.board_region))
        board_rgb = cv2.cvtColor(board_img, cv2.COLOR_BGR2RGB)
        objects = {}
        grid_size = (6, 7)
        cell_width = self.board_region[2] // grid_size[0]
        cell_height = self.board_region[3] // grid_size[1]

        # Define RGB colors
        grid_cell_bg_color = (251, 246, 245) # each grid cell bg
        grid_cell_border_color = (255, 231, 239) # each grid cell edge
        whole_grid_border_color = (250, 240, 239) # the grid edge
        additional_border_color = (0, 0, 0) # the most outer edge

        for grid_x in range(grid_size[0]):
            for grid_y in range(grid_size[1]):
                cell_x_start = grid_x * cell_width
                cell_y_start = grid_y * cell_height
                cell_x_end = cell_x_start + cell_width
                cell_y_end = cell_y_start + cell_height

                cell_region = board_rgb[cell_y_start:cell_y_end, cell_x_start:cell_x_end]

                grid_index = grid_y * grid_size[0] + grid_x
                cv2.putText(board_img, str(grid_index), (cell_x_start + 5, cell_y_start + 20), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                cv2.rectangle(board_img, (cell_x_start, cell_y_start), 
                            (cell_x_end, cell_y_end), (0, 255, 0), 2)
                
                detected = False
                for obj_name, template in self.templates.items():
                    detected = False
                    for scale in np.linspace(0.5, 1.5, 20):
                        resized_template = cv2.resize(template, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
                        matches = cv2.matchTemplate(cell_region, resized_template, cv2.TM_CCOEFF_NORMED)
                        _, max_val, _, max_loc = cv2.minMaxLoc(matches)
                        print(f"Grid ({grid_x}, {grid_y}) - {obj_name} match value: {max_val:.2f} at scale {scale:.2f}")  # Debug
                        if max_val >= 0.7:
                            objects[(grid_x, grid_y)] = (max_loc, obj_name)
                            cv2.putText(board_img, obj_name, (cell_x_start + 5, cell_y_start + 40), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors[obj_name], 1)
                            detected = True
                            break
                    if detected:
                        break
                
                if not detected:
                    objects[(grid_x, grid_y)] = (None, 'unknown')
                    cv2.rectangle(board_img, (cell_x_start, cell_y_start), 
                                  (cell_x_end, cell_y_end), (0, 255, 0), 2)

        if len(objects) != 42:
            print(f"Expected 42 objects, but found {len(objects)}")
            return {}

        cv2.imshow('Detected Objects', board_img)
        cv2.waitKey(1)
        cv2.imwrite("debug_detected_objects.png", board_img)
        print(f"Detected {len(objects)} unique objects")
        return objects

    def find_match(self, objects):
        sorted_objects = sorted(objects.items())
        for (grid_x, grid_y), (pt, obj_name) in sorted_objects:
            if (grid_x + 1, grid_y) in objects and objects[(grid_x + 1, grid_y)][1] == obj_name:
                return pt, objects[(grid_x + 1, grid_y)][0]
            if (grid_x, grid_y + 1) in objects and objects[(grid_x, grid_y + 1)][1] == obj_name:
                return pt, objects[(grid_x, grid_y + 1)][0]
        return None

    def make_move(self, start, end):
        start_x, start_y = start
        end_x, end_y = end
        start_center_x = self.board_region[0] + start_x + 25
        start_center_y = self.board_region[1] + start_y + 50
        end_center_x = self.board_region[0] + end_x + 25
        end_center_y = self.board_region[1] + end_y + 50
        pyautogui.moveTo(start_center_x, start_center_y)
        pyautogui.dragTo(end_center_x, end_center_y, button='left')

    def play(self):
        while self.running:
            print("Capturing game board...")
            objects = self.locate_objects()
            if len(objects) != 42:
                print(f"Expected 42 objects, but found {len(objects)}")
                break
            print(f"Found {len(objects)} objects")
            match = self.find_match(objects)
            if match:
                print(f"Found match: {match}")
                self.make_move(*match)
            else:
                print("No moves found")
                break
            time.sleep(2)

    def setup_key_listener(self):
        def on_press(key):
            try:
                if key.char == '0':
                    self.running = False
                    print("Quitting...")
            except AttributeError:
                pass

        listener = keyboard.Listener(on_press=on_press)
        listener.start()

if __name__ == "__main__":
    bot = Match3Solver()
    bot.play()
