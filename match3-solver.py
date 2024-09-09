import pyautogui
import cv2
import numpy as np
import time
from pynput import keyboard
import math
import pydirectinput

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
        # Load and verify template images for each object
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

    def detect_game_board(self, likely_top_left=(745, 186), likely_bottom_right=(1366, 906)):
        # If likely_bottom_right is not provided, use the full screen
        if likely_bottom_right is None:
            likely_bottom_right = pyautogui.size()

        # Calculate the region of interest
        roi_left, roi_top = likely_top_left
        roi_width = likely_bottom_right[0] - likely_top_left[0]
        roi_height = likely_bottom_right[1] - likely_top_left[1]

        # Capture only the region of interest
        screen = np.array(pyautogui.screenshot(region=(roi_left, roi_top, roi_width, roi_height)))
        screen_rgb = cv2.cvtColor(screen, cv2.COLOR_BGR2RGB)

        template = cv2.imread('board_template.png')
        template_rgb = cv2.cvtColor(template, cv2.COLOR_BGR2RGB)

        res = cv2.matchTemplate(screen_rgb, template_rgb, cv2.TM_CCOEFF_NORMED)
        _, _, _, max_loc = cv2.minMaxLoc(res)

        # Adjust the coordinates to account for the ROI offset
        top_left = (roi_left + max_loc[0], roi_top + max_loc[1])
        h, w = template_rgb.shape[:2]

        return (top_left[0], top_left[1], w, h)


    def locate_objects(self):
        # Capture the game board region
        board_img = np.array(pyautogui.screenshot(region=self.board_region))
        board_rgb = cv2.cvtColor(board_img, cv2.COLOR_BGR2RGB)
        objects = {}

        grid_size = (6, 7)
        cell_width = self.board_region[2] // grid_size[0]
        cell_height = self.board_region[3] // grid_size[1]

        # Define RGB colors for grid cell backgrounds and borders
        grid_cell_bg_color = (251, 246, 245)
        grid_cell_border_color = (255, 231, 239)
        whole_grid_border_color = (250, 240, 239)
        additional_border_color = (0, 0, 0)

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
                        if max_val >= 0.7:
                            template_width, template_height = resized_template.shape[1::-1]
                            obj_center = (cell_x_start + max_loc[0] + template_width // 2, cell_y_start + max_loc[1] + template_height // 2)
                            objects[(grid_x, grid_y)] = (obj_center, obj_name)
                            cv2.putText(board_img, obj_name, (cell_x_start + 5, cell_y_start + 40), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors[obj_name], 1)
                            
                            detected = True
                            break
                    if detected:
                        break
                
                if not detected:
                    # If no object detected in the cell, mark it as unknown
                    objects[(grid_x, grid_y)] = (None, 'unknown')

        if len(objects) != 42:
            print(f"Expected 42 objects, but found {len(objects)}")
            return {}

        cv2.imshow('Detected Objects', board_img)
        cv2.waitKey(1)
        cv2.imwrite("debug_detected_objects.png", board_img)
        print(f"Detected {len(objects)} unique objects")
        return objects

    def find_match(self, objects):
        # Initialize a 2D list to store object names for readability
        grid_size = (6, 7)
        readable_grid = [['unknown' for _ in range(grid_size[1])] for _ in range(grid_size[0])]

        # Populate the readable grid with object names
        for (grid_col, grid_row), (_, obj_name) in objects.items():
            readable_grid[grid_col][grid_row] = obj_name

        # Print the readable grid
        for row in range(grid_size[1]):
            print(' '.join(readable_grid[col][row] for col in range(grid_size[0])))

        def is_valid(x, y):
            return 0 <= x < grid_size[0] and 0 <= y < grid_size[1]

        def find_chain_length(x, y, dx, dy):
            length = 1
            obj_name = readable_grid[x][y]
            nx, ny = x + dx, y + dy
            while is_valid(nx, ny) and readable_grid[nx][ny] == obj_name:
                length += 1
                nx += dx
                ny += dy
            return length

        def evaluate_swap(x1, y1, x2, y2):
            # Swap the objects
            readable_grid[x1][y1], readable_grid[x2][y2] = readable_grid[x2][y2], readable_grid[x1][y1]
            
            # Check for matches
            max_length = 0
            for dx, dy in [(1, 0), (0, 1)]:
                max_length = max(max_length, find_chain_length(x1, y1, dx, dy) + find_chain_length(x1, y1, -dx, -dy) - 1)
                max_length = max(max_length, find_chain_length(x2, y2, dx, dy) + find_chain_length(x2, y2, -dx, -dy) - 1)
            
            # Swap back
            readable_grid[x1][y1], readable_grid[x2][y2] = readable_grid[x2][y2], readable_grid[x1][y1]
            
            return max_length

        def find_all_swaps():
            swaps = []
            for x in range(grid_size[0]):
                for y in range(grid_size[1]):
                    if is_valid(x + 1, y):
                        swaps.append(((x, y), (x + 1, y)))
                    if is_valid(x, y + 1):
                        swaps.append(((x, y), (x, y + 1)))
            return swaps

        def backtrack(swaps):
            best_match = None
            best_length = 0
            for (x1, y1), (x2, y2) in swaps:
                swap_length = evaluate_swap(x1, y1, x2, y2)
                if swap_length >= 3 and swap_length > best_length:
                    best_length = swap_length
                    best_match = ((x1, y1), (x2, y2))
            return best_match

        swaps = find_all_swaps()
        best_match = backtrack(swaps)
        return best_match

    # Function for smoothly moving the cursor and holding down the mouse button until the movement is completed
    @staticmethod
    def distance(point1, point2):
        return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

    def make_move(self, start, end):
        # Verify that the board region is correctly identified
        if self.board_region is None or len(self.board_region) != 4:
            raise ValueError("Board region is not correctly defined.")
        
        # Unpack board_region
        top_left_x, top_left_y, board_width, board_height = self.board_region
        
        # Debug: Print out the board region
        print(f"Board region: {self.board_region}")
        
        # Grid size (assumed 6x7, adjust if different)
        grid_columns = 6
        grid_rows = 7
        
        # Calculate cell dimensions
        cell_width = board_width // grid_columns
        cell_height = board_height // grid_rows
        
        # Debug: Print out cell dimensions
        print(f"Cell width: {cell_width}, Cell height: {cell_height}")
        
        # Extract start and end positions
        start_x, start_y = start
        end_x, end_y = end
        
        # Calculate the center of the starting grid cell
        start_center_x = top_left_x + (start_x * cell_width) + (cell_width // 2)
        start_center_y = top_left_y + (start_y * cell_height) + (cell_height // 2)
        
        # Calculate the center of the ending grid cell
        end_center_x = top_left_x + (end_x * cell_width) + (cell_width // 2)
        end_center_y = top_left_y + (end_y * cell_height) + (cell_height // 2)
        
        # Debugging output
        print(f"Start Center: ({start_center_x}, {start_center_y})")
        print(f"End Center: ({end_center_x}, {end_center_y})")
        
        # Adjust based on observed discrepancies
        # You may need to add or subtract pixels to fine-tune the position.
        adjustment_x = 0
        adjustment_y = 0
        start_center_x += adjustment_x
        start_center_y += adjustment_y
        end_center_x += adjustment_x
        end_center_y += adjustment_y
        
        # move to starting pos
        pydirectinput.moveTo(start_center_x, start_center_y)

        # start holding
        pydirectinput.mouseDown()
        
        # drag to the end pos
        pydirectinput.moveTo(end_center_x, end_center_y, duration=0.3)

        # dont let go until we arrive
        while Match3Solver.distance(pydirectinput.position(), (end_center_x, end_center_y)) > 2:  # Ignore distance to endpoint
            time.sleep(0.2)
        # let go
        time.sleep(0.1)
        pydirectinput.mouseUp()

    def wait_for_board_stability(self, max_wait_time=10):
        print("Waiting for board to stabilize...")
        start_time = time.time()
        last_board = None

        while time.time() - start_time < max_wait_time:
            current_board = np.array(pyautogui.screenshot(region=self.board_region))
            
            if last_board is not None:
                # Compare the current board with the last board
                difference = np.sum(np.abs(current_board - last_board))
                if difference == 0:
                    print("Board has stabilized.")
                    return True
            
            last_board = current_board
            time.sleep(0.2)  # Check every 0.2 seconds

        print("Board did not stabilize within the maximum wait time.")
        return False

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
                # Wait for the board to stop moving before continuing
                self.wait_for_board_stability()
            else:
                print("No moves found")
                break

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
