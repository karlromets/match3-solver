import pyautogui
import cv2
import numpy as np
import time
from pynput import keyboard
import math
import pydirectinput
import pyautogui
import os
from joblib import Parallel, delayed

def process_cell(args):
    self, grid_x, grid_y, cell_region, scaled_templates = args
    for obj_name, templates in scaled_templates.items():
        for template in templates:
            matches = cv2.matchTemplate(cell_region, template, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(matches)
            if max_val >= 0.85:
                template_width, template_height = template.shape[1::-1]
                obj_center = (max_loc[0] + template_width // 2, 
                              max_loc[1] + template_height // 2)
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
        self.locate_times = []
        self.match_times = []
        self.move_times = []
        self.total_moves = 0
        self.start_time = time.time()

    def load_templates(self):
        start_time = time.time()
        templates = {
            'object1': cv2.imread('object1_template.png', cv2.IMREAD_GRAYSCALE),
            'object2': cv2.imread('object2_template.png', cv2.IMREAD_GRAYSCALE),
            'object3': cv2.imread('object3_template.png', cv2.IMREAD_GRAYSCALE),
            'object4': cv2.imread('object4_template.png', cv2.IMREAD_GRAYSCALE),
            'object5': cv2.imread('object5_template.png', cv2.IMREAD_GRAYSCALE),
            'object6': cv2.imread('object6_template.png', cv2.IMREAD_GRAYSCALE)
        }
        for name, template in templates.items():
            if template is None:
                print(f"Failed to load template: {name}")

        print(f"Time to load templates: {time.time() - start_time:.2f} seconds")
        return templates

    def detect_game_board(self, likely_top_left=(817, 289), likely_bottom_right=(1294, 843)):
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
        screen_gray = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)

        template = cv2.imread('board_template.png', cv2.IMREAD_GRAYSCALE)

        res = cv2.matchTemplate(screen_gray, template, cv2.TM_CCOEFF_NORMED)
        _, _, _, max_loc = cv2.minMaxLoc(res)

        # Adjust the coordinates to account for the ROI offset
        top_left = (roi_left + max_loc[0], roi_top + max_loc[1])
        h, w = template.shape[:2]


        return (top_left[0], top_left[1], w, h)


    def locate_objects(self, known_objects=None):
        start_time = time.time()
        # Capture the game board region
        board_img = np.array(pyautogui.screenshot(region=self.board_region))
        board_gray = cv2.cvtColor(board_img, cv2.COLOR_BGR2GRAY)
        objects = {}

        grid_size = (6, 7)
        cell_width = self.board_region[2] // grid_size[0]
        cell_height = self.board_region[3] // grid_size[1]

        # Pre-compute scaled templates with fewer scales
        scaled_templates = {}
        for obj_name, template in self.templates.items():
            scaled_templates[obj_name] = [cv2.resize(template, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR) 
                                        for scale in np.linspace(0.9, 1.1, 3)]

        cell_regions = {}
        for x in range(grid_size[0]):
            for y in range(grid_size[1]):
                cell_x_start = x * cell_width
                cell_y_start = y * cell_height
                cell_x_end = cell_x_start + cell_width
                cell_y_end = cell_y_start + cell_height
                cell_regions[(x, y)] = board_gray[cell_y_start:cell_y_end, cell_x_start:cell_x_end]
        
        results = Parallel(n_jobs=-1)(
            delayed(process_cell)((self, x, y, cell_regions[(x, y)], scaled_templates))
            for x in range(grid_size[0])
            for y in range(grid_size[1])
            if not known_objects or (x, y) not in known_objects
        )

        objects.update(dict(results))

                # Save the debug image with detected objects
        for (grid_x, grid_y), (pt, obj_name) in objects.items():
            cell_x_start = grid_x * cell_width
            cell_y_start = grid_y * cell_height
            cell_x_end = cell_x_start + cell_width
            cell_y_end = cell_y_start + cell_height
            cv2.putText(board_img, obj_name, (cell_x_start + 5, cell_y_start + 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors.get(obj_name, (255, 255, 255)), 1)
            cv2.rectangle(board_img, (cell_x_start, cell_y_start), 
                          (cell_x_end, cell_y_end), (0, 255, 0), 2)

        cv2.imshow('Detected Objects', board_img)
        cv2.waitKey(1)
        cv2.imwrite("debug_detected_objects.png", board_img)

        locate_time = time.time() - start_time
        self.locate_times.append(locate_time)
        return objects

    def find_match(self, objects):
        start_time = time.time()
        grid_size = (6, 7)
        readable_grid = [['unknown' for _ in range(grid_size[1])] for _ in range(grid_size[0])]

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
            readable_grid[x1][y1], readable_grid[x2][y2] = readable_grid[x2][y2], readable_grid[x1][y1]
            
            matches = []
            for dx, dy in [(1, 0), (0, 1)]:
                for x, y in [(x1, y1), (x2, y2)]:
                    length = find_chain_length(x, y, dx, dy) + find_chain_length(x, y, -dx, -dy) - 1
                    if length >= 3:
                        matches.append((length, y, dx == 0))  # (length, row, is_vertical)
            
            readable_grid[x1][y1], readable_grid[x2][y2] = readable_grid[x2][y2], readable_grid[x1][y1]
            
            return matches

        def score_matches(matches):
            if not matches:
                return -1
            combo = len(matches)
            max_length = max(match[0] for match in matches)
            is_vertical = any(match[2] for match in matches)
            lowest_row = max(match[1] for match in matches)
            
            # Prioritize: 1. Combos, 2. Length, 3. Vertical, 4. Lowest row
            return (combo * 1000000) + (max_length * 10000) + (is_vertical * 100) + (lowest_row)

        best_swap = None
        best_score = -1

        for x in range(grid_size[0]):
            for y in range(grid_size[1]):
                for dx, dy in [(1, 0), (0, 1)]:
                    nx, ny = x + dx, y + dy
                    if is_valid(nx, ny):
                        matches = evaluate_swap(x, y, nx, ny)
                        score = score_matches(matches)
                        if score > best_score:
                            best_score = score
                            best_swap = ((x, y), (nx, ny))

        match_time = time.time() - start_time
        self.match_times.append(match_time)
        return best_swap

    # Function for smoothly moving the cursor and holding down the mouse button until the movement is completed
    @staticmethod
    def distance(point1, point2):
        return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

    def make_move(self, start, end):
        start_time = time.time()
        pydirectinput.PAUSE = 0.01
        
        # Unpack board_region (assuming it's correctly defined)
        top_left_x, top_left_y, board_width, board_height = self.board_region
        
        # Pre-calculate cell dimensions (can be done in __init__ if grid size is constant)
        cell_width = board_width // 6
        cell_height = board_height // 7
        
        # Calculate start and end positions in one step
        start_x, start_y = start
        end_x, end_y = end
        start_pos = (
            top_left_x + (start_x * cell_width) + (cell_width // 2),
            top_left_y + (start_y * cell_height) + (cell_height // 2)
        )
        end_pos = (
            top_left_x + (end_x * cell_width) + (cell_width // 2),
            top_left_y + (end_y * cell_height) + (cell_height // 2)
        )
        
        # Perform the mouse movement in one smooth action
        pydirectinput.moveTo(*start_pos)
        pydirectinput.mouseDown()
        pydirectinput.moveTo(*end_pos)
        pydirectinput.mouseUp()

        move_time = time.time() - start_time
        self.move_times.append(move_time)
        self.total_moves += 1

    def wait_for_board_stability(self, max_wait_time=10):
        function_start_time = time.time()  # Start timing the entire function execution
        start_time = time.time()
        last_board = None
        check_interval = 0.5  # Check every 0.5 seconds
        known_objects = {}

        while time.time() - start_time < max_wait_time:
            current_board = np.array(pyautogui.screenshot(region=self.board_region))
            
            if last_board is not None and np.array_equal(current_board, last_board):
                stability_time = time.time() - start_time
                function_duration = time.time() - function_start_time  # Calculate total function duration
                return True
            
            last_board = current_board

            # Check for popup every 1.5 seconds
            if time.time() - start_time >= 1.5:
                objects = self.locate_objects(known_objects)
                if len(objects) < 42:
                    print(f"Popup detected. Found {len(objects)} objects.")
                    known_objects = objects
                    match = self.find_match_outside_popup(objects)
                else:
                    match = self.find_match(objects)
                    
                # Try to find a match even if the popup is up
                if match:
                    self.make_move(*match)
                    start_time = time.time()  # Reset the timer
                    last_board = None
                    known_objects = {}
                else:
                    print("No moves found")

            time.sleep(check_interval)

        print("Board did not stabilize within the maximum wait time.")
        function_duration = time.time() - function_start_time  # Calculate total function duration
        return False

    def find_match_outside_popup(self, objects):
        start_time = time.time()
        grid_size = (6, 7)
        readable_grid = [['unknown' for _ in range(grid_size[1])] for _ in range(grid_size[0])]

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
            readable_grid[x1][y1], readable_grid[x2][y2] = readable_grid[x2][y2], readable_grid[x1][y1]
            
            max_length = 0
            total_matches = 0
            for dx, dy in [(1, 0), (0, 1)]:
                for x, y in [(x1, y1), (x2, y2)]:
                    length = find_chain_length(x, y, dx, dy) + find_chain_length(x, y, -dx, -dy) - 1
                    if length >= 3:
                        max_length = max(max_length, length)
                        total_matches += 1
            
            readable_grid[x1][y1], readable_grid[x2][y2] = readable_grid[x2][y2], readable_grid[x1][y1]
            
            return max_length, total_matches

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

            for swap in swaps:
                (x1, y1), (x2, y2) = swap
                swap_length, total_matches = evaluate_swap(x1, y1, x2, y2)
                score = swap_length + total_matches * 2  # Prioritize multiple matches
                if swap_length >= 3 and score > best_score:
                    best_score = score
                    best_match = ((x1, y1), (x2, y2))

            return best_match

        swaps = find_all_swaps()
        best_match = backtrack(swaps)
        match_time = time.time() - start_time
        self.match_times.append(match_time)
        return best_match

    def print_averages(self):
        os.system('cls' if os.name == 'nt' else 'clear')  # Clear the terminal
        avg_locate = sum(self.locate_times) / len(self.locate_times) if self.locate_times else 0
        avg_match = sum(self.match_times) / len(self.match_times) if self.match_times else 0
        avg_move = sum(self.move_times) / len(self.move_times) if self.move_times else 0
        
        # Calculate avg_total as the sum of the other averages
        avg_total = avg_locate + avg_match + avg_move

        current_time = time.time() - self.start_time

        print(f"Avg. time to locate objects: {avg_locate:.2f}s")
        print(f"Avg. time to find match: {avg_match:.2f}s")
        print(f"Avg. time to make move: {avg_move:.2f}s")
        print(f"Avg. total time: {avg_total:.2f}s")
        print(f"Total moves made: {self.total_moves}")
        print(f"Current game time: {current_time:.2f}s")

    def play(self):
        while self.running:
            objects = self.locate_objects()
            if len(objects) != 42:
                print(f"Expected 42 objects, but found {len(objects)}")
                break
            match = self.find_match(objects)
            if match:
                self.print_averages()
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
    bot = Match3Solver()
    bot.play()