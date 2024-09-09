import pyautogui
import cv2
import numpy as np
import time
import threading
from pynput import keyboard
import math
import pydirectinput
import pyautogui
import multiprocessing

def process_cell(args):
    self, grid_x, grid_y, board_rgb, cell_width, cell_height, scaled_templates = args
    cell_x_start = grid_x * cell_width
    cell_y_start = grid_y * cell_height
    cell_x_end = cell_x_start + cell_width
    cell_y_end = cell_y_start + cell_height

    cell_region = board_rgb[cell_y_start:cell_y_end, cell_x_start:cell_x_end]

    for obj_name, templates in scaled_templates.items():
        for template in templates:
            matches = cv2.matchTemplate(cell_region, template, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(matches)
            if max_val >= 0.8:
                template_width, template_height = template.shape[1::-1]
                obj_center = (cell_x_start + max_loc[0] + template_width // 2, 
                              cell_y_start + max_loc[1] + template_height // 2)
                return (grid_x, grid_y), (obj_center, obj_name)

    return (grid_x, grid_y), (None, 'unknown')

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
            'object6': (255, 0, 255) # Magenta
        }
        self.running = True
        self.setup_key_listener()

    def load_templates(self):
        start_time = time.time()
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
            # else:
                # print(f"Loaded template: {name} - Shape: {template.shape}")
        print(f"Time to load templates: {time.time() - start_time:.2f} seconds")
        return templates

    def detect_game_board(self, likely_top_left=(745, 186), likely_bottom_right=(1366, 906)):
        start_time = time.time()
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

        print(f"Time to detect game board: {time.time() - start_time:.2f} seconds")
        return (top_left[0], top_left[1], w, h)


    def locate_objects(self):
        start_time = time.time()
        # Capture the game board region
        board_img = np.array(pyautogui.screenshot(region=self.board_region))
        board_rgb = cv2.cvtColor(board_img, cv2.COLOR_BGR2RGB)
        objects = {}

        grid_size = (6, 7)
        cell_width = self.board_region[2] // grid_size[0]
        cell_height = self.board_region[3] // grid_size[1]

        # Pre-compute scaled templates with fewer scales
        scaled_templates = {}
        for obj_name, template in self.templates.items():
            scaled_templates[obj_name] = [cv2.resize(template, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR) 
                                          for scale in np.linspace(0.9, 1.1, 3)]  # Reduced to 3 scales

        # Prepare arguments for multiprocessing
        args = [(self, x, y, board_rgb, cell_width, cell_height, scaled_templates) 
                for x in range(grid_size[0]) for y in range(grid_size[1])]

        # Use multiprocessing to process cells in parallel
        with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:  # Ensure full CPU utilization
            results = pool.map(process_cell, args)

        objects = dict(results)

        if len(objects) != 42:
            print(f"Expected 42 objects, but found {len(objects)}")
            return {}

        print(f"Time to locate objects: {time.time() - start_time:.2f} seconds")
        return objects

    def find_match(self, objects):
        start_time = time.time()
        # Initialize a 2D list to store object names for readability
        grid_size = (6, 7)
        readable_grid = [['unknown' for _ in range(grid_size[1])] for _ in range(grid_size[0])]

        # Populate the readable grid with object names
        for (grid_col, grid_row), (_, obj_name) in objects.items():
            readable_grid[grid_col][grid_row] = obj_name

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
            additional_matches = 0
            for dx, dy in [(1, 0), (0, 1)]:
                for x, y in [(x1, y1), (x2, y2)]:
                    length = find_chain_length(x, y, dx, dy) + find_chain_length(x, y, -dx, -dy) - 1
                    if length >= 3:
                        max_length = max(max_length, length)
                        additional_matches += 1 if length > 3 else 0
            
            # Swap back
            readable_grid[x1][y1], readable_grid[x2][y2] = readable_grid[x2][y2], readable_grid[x1][y1]
            
            return max_length, additional_matches

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
            best_score = 0
            lock = threading.Lock()

            def evaluate_and_update_best(swap):
                nonlocal best_match, best_score
                (x1, y1), (x2, y2) = swap
                swap_length, additional_matches = evaluate_swap(x1, y1, x2, y2)
                score = swap_length + additional_matches * 2  # Prioritize additional matches
                if swap_length >= 3 and score > best_score:
                    with lock:
                        if score > best_score:
                            best_score = score
                            best_match = ((x1, y1), (x2, y2))

            threads = []
            for swap in swaps:
                thread = threading.Thread(target=evaluate_and_update_best, args=(swap,))
                threads.append(thread)
                thread.start()

            for thread in threads:
                thread.join()

            return best_match

        swaps = find_all_swaps()
        best_match = backtrack(swaps)
        print(f"Time to find match: {time.time() - start_time:.2f} seconds")
        return best_match

    # Function for smoothly moving the cursor and holding down the mouse button until the movement is completed
    @staticmethod
    def distance(point1, point2):
        return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

    def make_move(self, start, end):
        start_time = time.time()
        # Verify that the board region is correctly identified
        if self.board_region is None or len(self.board_region) != 4:
            raise ValueError("Board region is not correctly defined.")
        
        # Unpack board_region
        top_left_x, top_left_y, board_width, board_height = self.board_region
        
        # Grid size (assumed 6x7, adjust if different)
        grid_columns = 6
        grid_rows = 7
        
        # Calculate cell dimensions
        cell_width = board_width // grid_columns
        cell_height = board_height // grid_rows
        
        # Extract start and end positions
        start_x, start_y = start
        end_x, end_y = end
        
        # Calculate the center of the starting grid cell
        start_center_x = top_left_x + (start_x * cell_width) + (cell_width // 2)
        start_center_y = top_left_y + (start_y * cell_height) + (cell_height // 2)
        
        # Calculate the center of the ending grid cell
        end_center_x = top_left_x + (end_x * cell_width) + (cell_width // 2)
        end_center_y = top_left_y + (end_y * cell_height) + (cell_height // 2)

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
        pydirectinput.moveTo(end_center_x, end_center_y)
        pydirectinput.mouseUp()

        print(f"Time to make move: {time.time() - start_time:.2f} seconds")

    def wait_for_board_stability(self, max_wait_time=10):
        start_time = time.time()
        # print("Waiting for board to stabilize...")
        start_time = time.time()
        last_board = None

        while time.time() - start_time < max_wait_time:
            current_board = np.array(pyautogui.screenshot(region=self.board_region))
            
            if last_board is not None:
                # Compare the current board with the last board
                difference = np.sum(np.abs(current_board - last_board))
                if difference == 0:
                    # print("Board has stabilized.")
                    print(f"Time waiting for board stability: {time.time() - start_time:.2f} seconds")
                    return True
            
            last_board = current_board
            time.sleep(0.05)  # Check every 0.05 seconds

        print("Board did not stabilize within the maximum wait time.")
        return False

    def play(self):
        while self.running:
            move_start_time = time.time()
            print("Starting new move...")
            objects = self.locate_objects()
            if len(objects) != 42:
                print(f"Expected 42 objects, but found {len(objects)}")
                break
            match = self.find_match(objects)
            if match:
                total_move_time = time.time() - move_start_time
                print(f"Total time for current move: {total_move_time:.2f} seconds")
                # print(f"Found match: {match}")
                self.make_move(*match)
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
    multiprocessing.freeze_support()  # This line is necessary for Windows
    bot = Match3Solver()
    bot.play()