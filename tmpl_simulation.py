import curses
import time
import threading
from datetime import datetime

class TMPLSimulation:
    def __init__(self, stdscr):
        self.stdscr = stdscr
        curses.curs_set(0)  # Hide cursor
        curses.start_color()
        curses.use_default_colors()
        
        # Simulation parameters
        self.rows = 3
        self.cols = 10
        self.N = 240  # Maximum saturation value
        
        # Initialize grids
        self.landscape = [
            [{'saturation': 0, 'processed': False} for _ in range(self.cols + 1)]
            for _ in range(self.rows + 1)
        ]
        self.visitors = [
            [{'present': False} for _ in range(self.cols + 1)]
            for _ in range(self.rows + 1)
        ]
        
        # Initial state (saturate first 5 cells in second row)
        for col in range(1, 6):
            self.landscape[2][col]['saturation'] = self.N

        # Cursor position
        self.cursor_row = 1
        self.cursor_col = 1

        # Control variables
        self.running = True
        self.start_time = datetime.now()
        
        # Lock for safe screen access
        self.screen_lock = threading.Lock()

        # Initialize log file
        self.log_filename = f'tmpl.log'
        self.last_logged_state = None
        open(self.log_filename, 'w').close()  # Clear/create the file
        self.log_state()  # Log initial state

    def log_state(self, event=None):
        current_state = [self.landscape[2][col]['saturation'] for col in range(6, self.cols + 1)]
        
        # Only log if state changed
        if current_state != self.last_logged_state:
            with open(self.log_filename, 'a') as f:
                f.write(f"{current_state}\n")
            self.last_logged_state = current_state.copy()

    def render(self):
        with self.screen_lock:
            self.stdscr.clear()
            
            # Display exposure time
            elapsed = datetime.now() - self.start_time
            hours = int(elapsed.total_seconds() // 3600)
            minutes = int((elapsed.total_seconds() % 3600) // 60)
            seconds = int(elapsed.total_seconds() % 60)
            self.stdscr.addstr(0, 0, f"Time: {hours:02d}:{minutes:02d}:{seconds:02d}")
            
            # Display landscape grid with values
            for row in range(1, self.rows + 1):
                for col in range(1, self.cols + 1):
                    saturation = self.landscape[row][col]['saturation']
                    # If N is reached, show filled cell
                    if saturation >= self.N:
                        value_str = "[░░░]"
                    else:
                        value_str = f"[{saturation:3d}]"
                    self.stdscr.addstr(row * 2, col * 6, value_str)
            
            # Empty line between grids
            empty_row = self.rows * 2 + 1
            
            # Display visitors grid
            for row in range(1, self.rows + 1):
                for col in range(1, self.cols + 1):
                    char = "[ * ]" if self.visitors[row][col]['present'] else "[   ]"
                    if row == self.cursor_row and col == self.cursor_col:
                        self.stdscr.attron(curses.A_REVERSE)
                        self.stdscr.addstr(empty_row + row, col * 6, char)
                        self.stdscr.attroff(curses.A_REVERSE)
                    else:
                        self.stdscr.addstr(empty_row + row, col * 6, char)
            
            # Display instructions
            base_instruction_row = empty_row + self.rows + 2
            self.stdscr.addstr(base_instruction_row, 0, "Controls:")
            self.stdscr.addstr(base_instruction_row + 1, 0, "Arrows - move cursor")
            self.stdscr.addstr(base_instruction_row + 2, 0, "Space - toggle presence")
            self.stdscr.addstr(base_instruction_row + 3, 0, "Q - quit simulation")
            
            # Display row 2 state (debug)
            if self.rows >= 2:
                row_state = [self.landscape[2][col]['saturation'] for col in range(6, self.cols + 1)]
                self.stdscr.addstr(base_instruction_row + 5, 0, f"Row 2: {row_state}")
            
            self.stdscr.refresh()

    def handle_saturation_overflow(self, row, col):
        if self.landscape[row][col]['saturation'] < self.N:
            return

        neighbors = [
            {'col': col - 1, 'direction': -1},
            {'col': col + 1, 'direction': 1}
        ]

        min_neighbor = None
        min_saturation = float('inf')

        for neighbor in neighbors:
            ncol = neighbor['col']
            if 1 <= ncol <= self.cols:
                if not self.landscape[row][ncol]['processed']:
                    saturation = self.landscape[row][ncol]['saturation']
                    if saturation < self.N and saturation < min_saturation:
                        min_saturation = saturation
                        min_neighbor = neighbor

        if min_neighbor:
            ncol = min_neighbor['col']
            self.landscape[row][ncol]['saturation'] += 1
            self.landscape[row][ncol]['processed'] = True
            self.handle_saturation_overflow(row, ncol)
            return

        for distance in range(2, self.cols + 1):
            for direction in [-1, 1]:
                next_col = col + (distance * direction)
                if 1 <= next_col <= self.cols:
                    if not self.landscape[row][next_col]['processed'] and \
                       self.landscape[row][next_col]['saturation'] < self.N:
                        self.landscape[row][next_col]['saturation'] += 1
                        self.landscape[row][next_col]['processed'] = True
                        self.handle_saturation_overflow(row, next_col)
                        return

        if all(self.landscape[row][c]['saturation'] >= self.N for c in range(1, self.cols + 1)):
            self.running = False

    def update_saturation(self):
        # Reset processed flags
        for row in range(1, self.rows + 1):
            for col in range(1, self.cols + 1):
                self.landscape[row][col]['processed'] = False

        changes_made = False
        for row in range(1, self.rows + 1):
            for col in range(1, self.cols + 1):
                if self.visitors[row][col]['present']:
                    if not self.landscape[row][col]['processed']:
                        if self.landscape[row][col]['saturation'] >= self.N:
                            self.handle_saturation_overflow(row, col)
                            changes_made = True
                        else:
                            self.landscape[row][col]['saturation'] += 1
                            self.landscape[row][col]['processed'] = True
                            changes_made = True

        # Log only when changes occur
        if changes_made:
            self.log_state()

    def update_loop(self):
        while self.running:
            self.update_saturation()
            time.sleep(0.333)

    def run(self):
        # Start update thread
        update_thread = threading.Thread(target=self.update_loop)
        update_thread.start()

        self.stdscr.nodelay(1)  # Non-blocking mode for getch()
        
        while self.running:
            self.render()
            
            try:
                key = self.stdscr.getch()
                if key == ord('q'):
                    self.running = False
                elif key == ord(' '):
                    self.visitors[self.cursor_row][self.cursor_col]['present'] = \
                        not self.visitors[self.cursor_row][self.cursor_col]['present']
                    # Log state when visitor is added/removed
                    self.log_state("Visitor state changed")
                elif key == curses.KEY_UP and self.cursor_row > 1:
                    self.cursor_row -= 1
                elif key == curses.KEY_DOWN and self.cursor_row < self.rows:
                    self.cursor_row += 1
                elif key == curses.KEY_LEFT and self.cursor_col > 1:
                    self.cursor_col -= 1
                elif key == curses.KEY_RIGHT and self.cursor_col < self.cols:
                    self.cursor_col += 1
            except curses.error:
                pass

            time.sleep(0.1)

        # Log final state
        self.log_state("Simulation ended")
        update_thread.join()

def main(stdscr):
    simulation = TMPLSimulation(stdscr)
    simulation.run()

if __name__ == "__main__":
    curses.wrapper(main)