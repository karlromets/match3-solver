# Match-3 Game Solver

This bot automates playing Match-3 style games by constantly looking for matches and making moves until it either can't find any more moves or you stop it.

## How it Works

1. **Find board**: Locate the game board on the screen.
2. **Locate objects**: Identify the objects on the board using template matching.
3. **Find the best match**: Analyze the board to find the best possible match.
4. **Make the swap**: Perform the swapping action to make the match.
5. **Wait for board stability**: Wait for the board to settle after the move.
6. **Repeat**: Continue from step 2 until no more moves are found or the bot is stopped.

## Getting Started

1. Install the necessary dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Get the board coordinates:
   - Run the `match3-getBoardPos.py` script.
   - Click the top-left and bottom-right corners of the game board.
   - Update the `detect_game_board()` function in `match3-solver.py` with the obtained coordinates.

3. Add template images:
   - Take screenshots of the individual objects in your specific game.
   - Save them as `object1_template.png`, `object2_template.png`, etc.

4. Adjust the grid size:
   - The bot assumes a 6x7 game board grid by default.
   - Modify the grid size in the code to match your game's layout.

5. Run the bot:
   ```
   python match3-solver.py
   ```

6. To stop the bot, press '0'.

## Limitations and Future Improvements

- The current method for waiting for board stability is primitive and could be improved.
- If objects are incorrectly detected, the solver may make invalid moves.
- The solver doesn't currently check if a move was successful or if the game has ended.

Feel free to contribute to the project and help address these limitations!

