import pygame
import sys
import time
import math
import random
from typing import List, Tuple, Optional, Dict

# ... [Previous Constants and Initialization Code] ...
# Initialize pygame
pygame.init()

# Constants
WINDOW_SIZE = 740
BOARD_SIZE = 600
GRID_SIZE = 3
CELL_SIZE = BOARD_SIZE // GRID_SIZE
ANIMATION_SPEED = 15  # Higher is faster
BOARD_PADDING = (WINDOW_SIZE - BOARD_SIZE) // 2

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (200, 200, 200)
LIGHT_GRAY = (230, 230, 230)
DARK_GRAY = (70, 70, 70)
RED = (255, 50, 50)
GREEN = (50, 255, 50)
BLUE = (50, 50, 255)
BACKGROUND_COLOR = (240, 240, 240)
LINE_COLOR = (50, 50, 50)
X_COLOR = (41, 128, 185)  # Blue
O_COLOR = (231, 76, 60)   # Red
BUTTON_COLOR = (52, 152, 219)
BUTTON_HOVER_COLOR = (41, 128, 185)
BUTTON_TEXT_COLOR = WHITE

# Fonts
FONT = pygame.font.SysFont('Arial', 24)
LARGE_FONT = pygame.font.SysFont('Arial', 36)
SMALL_FONT = pygame.font.SysFont('Arial', 18)

# Set up the display
screen = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE))
pygame.display.set_caption("Tic-Tac-Toe with AI")
clock = pygame.time.Clock()



class TicTacToe:
    def __init__(self):
        self.board = [[' ' for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]
        self.current_player = 'X'
        self.winning_line = None
        
    def reset(self):
        self.board = [[' ' for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]
        self.current_player = 'X'
        self.winning_line = None
    
    # ... [Other Methods Remain Unchanged] ...
    def is_valid_move(self, row: int, col: int) -> bool:
        """Check if a move is valid"""
        # Check if position is within bounds and empty
        return 0 <= row < GRID_SIZE and 0 <= col < GRID_SIZE and self.board[row][col] == ' '
    
    def make_move(self, row: int, col: int, player: str) -> bool:
        """Make a move on the board"""
        if self.is_valid_move(row, col):
            self.board[row][col] = player
            return True
        return False
    
    def get_valid_moves(self) -> List[Tuple[int, int]]:
        """Get all valid moves on the current board"""
        moves = []
        for i in range(GRID_SIZE):
            for j in range(GRID_SIZE):
                if self.board[i][j] == ' ':
                    moves.append((i, j))
        return moves
    
    def check_winner(self, store_line: bool = True) -> Optional[str]:
        """Check if there's a winner and store winning line if store_line is True"""
        winner = None
        winning_line = None
        
        # Check rows
        for i in range(GRID_SIZE):
            if self.board[i][0] == self.board[i][1] == self.board[i][2] != ' ':
                winning_line = [(i, 0), (i, 1), (i, 2)]
                winner = self.board[i][0]
                break
        
        if not winner:
            # Check columns
            for i in range(GRID_SIZE):
                if self.board[0][i] == self.board[1][i] == self.board[2][i] != ' ':
                    winning_line = [(0, i), (1, i), (2, i)]
                    winner = self.board[0][i]
                    break
        
        if not winner:
            # Check diagonals
            if self.board[0][0] == self.board[1][1] == self.board[2][2] != ' ':
                winning_line = [(0, 0), (1, 1), (2, 2)]
                winner = self.board[0][0]
            elif self.board[0][2] == self.board[1][1] == self.board[2][0] != ' ':
                winning_line = [(0, 2), (1, 1), (2, 0)]
                winner = self.board[0][2]
        
        if winner and store_line:
            self.winning_line = winning_line
        return winner
    
    # ... [Other Methods Remain Unchanged] ...
    def is_board_full(self) -> bool:
        """Check if the board is full"""
        for row in self.board:
            if ' ' in row:
                return False
        return True
    
    def game_over(self) -> bool:
        """Check if the game is over"""
        return self.check_winner() is not None or self.is_board_full()
    
    def switch_player(self):
        """Switch the current player"""
        self.current_player = 'O' if self.current_player == 'X' else 'X'
    
    def get_board_copy(self):
        """Return a copy of the current board"""
        return [row[:] for row in self.board]


class MinimaxAI:
    # ... [Previous MinimaxAI Code] ...
    def __init__(self, player: str, use_alpha_beta: bool = False):
        self.player = player  # 'X' or 'O'
        self.opponent = 'O' if player == 'X' else 'X'
        self.use_alpha_beta = use_alpha_beta
        self.nodes_evaluated = 0  # Counter for performance comparison
        self.evaluation_time = 0
    
    def reset_counter(self):
        """Reset the nodes evaluated counter"""
        self.nodes_evaluated = 0
    
    def get_best_move(self, game: TicTacToe) -> Tuple[int, int]:
        """Get the best move for the current board state"""
        self.reset_counter()
        start_time = time.time()
        
        best_score = float('-inf')
        best_move = None
        
        # If board is empty, make a random move (for variety)
        if all(game.board[i][j] == ' ' for i in range(GRID_SIZE) for j in range(GRID_SIZE)):
            return random.choice([(0, 0), (0, 2), (2, 0), (2, 2), (1, 1)])
        
        for move in game.get_valid_moves():
            row, col = move
            # Try this move
            game.board[row][col] = self.player
            
            # Calculate score using minimax
            if self.use_alpha_beta:
                score = self.minimax_alpha_beta(game, 0, False, float('-inf'), float('inf'))
            else:
                score = self.minimax(game, 0, False)
            
            # Undo the move
            game.board[row][col] = ' '
            
            # Update best score and move
            if score > best_score:
                best_score = score
                best_move = move
        
        end_time = time.time()
        self.evaluation_time = end_time - start_time
        
        return best_move
    
    def minimax(self, game: TicTacToe, depth: int, is_maximizing: bool) -> int:
        self.nodes_evaluated += 1
        
        # Check terminal states with store_line=False
        winner = game.check_winner(store_line=False)
        if winner == self.player:
            return 10 - depth
        elif winner == self.opponent:
            return depth - 10
        elif game.is_board_full():
            return 0
        
        # ... [Rest of minimax Code] ...
        if is_maximizing:
            best_score = float('-inf')
            for move in game.get_valid_moves():
                row, col = move
                game.board[row][col] = self.player
                score = self.minimax(game, depth + 1, False)
                game.board[row][col] = ' '
                best_score = max(score, best_score)
            return best_score
        else:
            best_score = float('inf')
            for move in game.get_valid_moves():
                row, col = move
                game.board[row][col] = self.opponent
                score = self.minimax(game, depth + 1, True)
                game.board[row][col] = ' '
                best_score = min(score, best_score)
            return best_score
    
    def minimax_alpha_beta(self, game: TicTacToe, depth: int, is_maximizing: bool, 
                          alpha: float, beta: float) -> int:
        self.nodes_evaluated += 1
        
        # Check terminal states with store_line=False
        winner = game.check_winner(store_line=False)
        if winner == self.player:
            return 10 - depth
        elif winner == self.opponent:
            return depth - 10
        elif game.is_board_full():
            return 0
        
        # ... [Rest of minimax_alpha_beta Code] ...
        if is_maximizing:
            best_score = float('-inf')
            for move in game.get_valid_moves():
                row, col = move
                game.board[row][col] = self.player
                score = self.minimax_alpha_beta(game, depth + 1, False, alpha, beta)
                game.board[row][col] = ' '
                best_score = max(score, best_score)
                alpha = max(alpha, best_score)
                if beta <= alpha:
                    break  # Beta cutoff
            return best_score
        else:
            best_score = float('inf')
            for move in game.get_valid_moves():
                row, col = move
                game.board[row][col] = self.opponent
                score = self.minimax_alpha_beta(game, depth + 1, True, alpha, beta)
                game.board[row][col] = ' '
                best_score = min(score, best_score)
                beta = min(beta, best_score)
                if beta <= alpha:
                    break  # Alpha cutoff
            return best_score


class GameGUI:
    # ... [Previous GameGUI Code] ...
    def __init__(self):
        self.game = TicTacToe()
        self.animation = Animation()
        self.game_state = "menu"  # menu, playing, game_over
        self.ai_thinking = False
        self.ai_move = None
        self.ai = None
        self.human_player = None
        self.ai_player = None
        self.performance_stats = {"nodes": 0, "time": 0}
        
        # Define buttons
        btn_width, btn_height = 300, 50
        center_x = WINDOW_SIZE // 2
        
        self.menu_buttons = [
            Button(center_x - btn_width // 2, 250, btn_width, btn_height, 
                   "Human vs AI (Standard)", lambda: self.start_game(False)),
            Button(center_x - btn_width // 2, 320, btn_width, btn_height, 
                   "Human vs AI (Alpha-Beta)", lambda: self.start_game(True)),
            Button(center_x - btn_width // 2, 390, btn_width, btn_height, 
                   "Compare Algorithms", self.compare_algorithms)
        ]
        
        self.game_buttons = [
            Button(center_x - btn_width // 2, WINDOW_SIZE - 70, btn_width, btn_height, 
                   "Back to Menu", self.back_to_menu)
        ]
    
    def start_game(self, use_alpha_beta=False):
        """Start a new game"""
        self.game.reset()
        self.animation.reset_winner_animation()
        self.game_state = "playing"
        self.human_player = 'X'
        self.ai_player = 'O'
        self.ai = MinimaxAI(self.ai_player, use_alpha_beta=use_alpha_beta)
        return True
    
    def back_to_menu(self):
        """Return to the main menu"""
        self.game_state = "menu"
        return True
    
    def compare_algorithms(self):
        """Compare the performance of both algorithms"""
        self.game_state = "comparing"
        self.game.reset()
        
        # Compare standard Minimax
        std_ai = MinimaxAI('X', use_alpha_beta=False)
        start_time = time.time()
        std_ai.get_best_move(self.game)
        std_time = time.time() - start_time
        std_nodes = std_ai.nodes_evaluated
        
        # Compare Alpha-Beta Pruning
        self.game.reset()
        ab_ai = MinimaxAI('X', use_alpha_beta=True)
        start_time = time.time()
        ab_ai.get_best_move(self.game)
        ab_time = time.time() - start_time
        ab_nodes = ab_ai.nodes_evaluated
        
        # Store results
        self.performance_stats = {
            "standard": {"nodes": std_nodes, "time": std_time},
            "alpha_beta": {"nodes": ab_nodes, "time": ab_time}
        }
        
        # Reset for visual clarity
        self.game.reset()
        return True
    
    def handle_cell_click(self, row, col):
        """Handle clicking on a cell"""
        if self.game_state != "playing" or self.ai_thinking or self.game.game_over():
            return
        
        if self.game.current_player == self.human_player and self.game.make_move(row, col, self.human_player):
            # Animate human move
            self.animation.add_piece_animation(row, col, self.human_player)
            
            # Check for game over
            if self.game.game_over():
                return
            
            # Switch player
            self.game.switch_player()
            
            # AI's turn
            self.ai_thinking = True
    
    def process_ai_move(self):
        """Process the AI's move"""
        if self.ai_thinking:
            # Get AI move
            row, col = self.ai.get_best_move(self.game)
            self.game.make_move(row, col, self.ai_player)
            
            # Store performance stats
            self.performance_stats = {
                "nodes": self.ai.nodes_evaluated,
                "time": self.ai.evaluation_time
            }
            
            # Add animation
            self.animation.add_piece_animation(row, col, self.ai_player)
            
            # Check winner after AI move
            if self.game.check_winner():
                self.animation.reset_winner_animation()
            
            # Switch player
            self.game.switch_player()
            
            # AI thinking done
            self.ai_thinking = False
    
    def draw_board(self, screen):
        """Draw the game board"""
        # Draw board background
        board_rect = pygame.Rect(BOARD_PADDING, BOARD_PADDING, BOARD_SIZE, BOARD_SIZE)
        pygame.draw.rect(screen, WHITE, board_rect)
        pygame.draw.rect(screen, BLACK, board_rect, 2)
        
        # Draw grid lines
        for i in range(1, GRID_SIZE):
            # Vertical lines
            pygame.draw.line(
                screen, LINE_COLOR,
                (BOARD_PADDING + i * CELL_SIZE, BOARD_PADDING),
                (BOARD_PADDING + i * CELL_SIZE, BOARD_PADDING + BOARD_SIZE),
                3
            )
            # Horizontal lines
            pygame.draw.line(
                screen, LINE_COLOR,
                (BOARD_PADDING, BOARD_PADDING + i * CELL_SIZE),
                (BOARD_PADDING + BOARD_SIZE, BOARD_PADDING + i * CELL_SIZE),
                3
            )
        
        # Draw X's and O's
        for row in range(GRID_SIZE):
            for col in range(GRID_SIZE):
                x = BOARD_PADDING + col * CELL_SIZE + CELL_SIZE // 2
                y = BOARD_PADDING + row * CELL_SIZE + CELL_SIZE // 2
                
                if self.game.board[row][col] == 'X':
                    # Draw X (already animated)
                    line_length = int(CELL_SIZE * 0.8)
                    line_width = 12
                    
                    pygame.draw.line(
                        screen, X_COLOR,
                        (x - line_length // 2, y - line_length // 2),
                        (x + line_length // 2, y + line_length // 2),
                        line_width
                    )
                    pygame.draw.line(
                        screen, X_COLOR,
                        (x + line_length // 2, y - line_length // 2),
                        (x - line_length // 2, y + line_length // 2),
                        line_width
                    )
                    
                elif self.game.board[row][col] == 'O':
                    # Draw O (already animated)
                    radius = int(CELL_SIZE * 0.4)
                    line_width = 10
                    
                    pygame.draw.circle(screen, O_COLOR, (x, y), radius, line_width)
        
        # Draw active animations
        self.animation.draw_animations(screen)
        
        # Draw winning line if game is over
        if self.game.winning_line:
            self.animation.draw_winning_line(screen, self.game.winning_line)
    
    def draw_status(self, screen):
        """Draw game status information"""
        # Create status bar
        status_rect = pygame.Rect(0, WINDOW_SIZE - 150, WINDOW_SIZE, 80)
        pygame.draw.rect(screen, LIGHT_GRAY, status_rect)
        pygame.draw.line(screen, GRAY, (0, WINDOW_SIZE - 150), (WINDOW_SIZE, WINDOW_SIZE - 150), 2)
        
        # Draw status text
        if self.game_state == "playing":
            # Show current player
            if not self.game.game_over():
                current_player = "Your" if self.game.current_player == self.human_player else "AI's"
                player_text = f"{current_player} turn"
                text_surface = FONT.render(player_text, True, BLACK)
                screen.blit(text_surface, (20, WINDOW_SIZE - 140))
            
            # Show winner or draw
            if self.game.game_over():
                winner = self.game.check_winner()
                if winner:
                    winner_text = "You win!" if winner == self.human_player else "AI wins!"
                    text_surface = LARGE_FONT.render(winner_text, True, GREEN if winner == self.human_player else RED)
                else:
                    text_surface = LARGE_FONT.render("It's a draw!", True, BLUE)
                
                text_rect = text_surface.get_rect(center=(WINDOW_SIZE // 2, WINDOW_SIZE - 120))
                screen.blit(text_surface, text_rect)
            
            # Show AI info if available
            if "nodes" in self.performance_stats and self.ai:
                algorithm = "Alpha-Beta Pruning" if self.ai.use_alpha_beta else "Standard Minimax"
                nodes_text = f"Algorithm: {algorithm} | Nodes evaluated: {self.performance_stats['nodes']}"
                time_text = f"Evaluation time: {self.performance_stats['time']:.4f}s"
                
                nodes_surface = SMALL_FONT.render(nodes_text, True, DARK_GRAY)
                time_surface = SMALL_FONT.render(time_text, True, DARK_GRAY)
                
                screen.blit(nodes_surface, (20, WINDOW_SIZE - 100))
                screen.blit(time_surface, (20, WINDOW_SIZE - 80))
    
    def draw_menu(self, screen):
        """Draw the main menu"""
        # Draw title
        title_text = LARGE_FONT.render("Tic-Tac-Toe with Minimax AI", True, BLACK)
        title_rect = title_text.get_rect(center=(WINDOW_SIZE // 2, 150))
        screen.blit(title_text, title_rect)
        
        # Draw menu buttons
        for button in self.menu_buttons:
            button.draw(screen)
    
    def draw_comparison_results(self, screen):
        """Draw the algorithm comparison results"""
        # Draw title
        title_text = LARGE_FONT.render("Algorithm Comparison", True, BLACK)
        title_rect = title_text.get_rect(center=(WINDOW_SIZE // 2, 100))
        screen.blit(title_text, title_rect)
        
        # Get stats
        if "standard" in self.performance_stats and "alpha_beta" in self.performance_stats:
            std_nodes = self.performance_stats["standard"]["nodes"]
            std_time = self.performance_stats["standard"]["time"]
            ab_nodes = self.performance_stats["alpha_beta"]["nodes"]
            ab_time = self.performance_stats["alpha_beta"]["time"]
            
            # Calculate improvement percentages
            nodes_improvement = ((std_nodes - ab_nodes) / std_nodes) * 100 if std_nodes > 0 else 0
            time_improvement = ((std_time - ab_time) / std_time) * 100 if std_time > 0 else 0
            
            # Create text
            lines = [
                f"Standard Minimax:",
                f"   Nodes evaluated: {std_nodes}",
                f"   Evaluation time: {std_time:.4f}s",
                "",
                f"Alpha-Beta Pruning:",
                f"   Nodes evaluated: {ab_nodes}",
                f"   Evaluation time: {ab_time:.4f}s",
                "",
                f"Improvement with Alpha-Beta:",
                f"   Nodes reduced: {nodes_improvement:.1f}%",
                f"   Time saved: {time_improvement:.1f}%"
            ]
            
            # Draw text
            y = 180
            for line in lines:
                text = FONT.render(line, True, BLACK)
                text_rect = text.get_rect(center=(WINDOW_SIZE // 2, y))
                screen.blit(text, text_rect)
                y += 40
        
        # Draw back button
        for button in self.game_buttons:
            button.draw(screen)
    
    def draw(self, screen):
        """Draw the game"""
        # Clear screen
        screen.fill(BACKGROUND_COLOR)
        
        if self.game_state == "menu":
            self.draw_menu(screen)
        elif self.game_state == "comparing":
            self.draw_comparison_results(screen)
        else:  # playing
            # Draw the board
            self.draw_board(screen)
            
            # Draw status information
            self.draw_status(screen)
            
            # Draw game buttons
            for button in self.game_buttons:
                button.draw(screen)
    
    def handle_events(self):
        """Handle pygame events"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            
            # Handle mouse motion for button hover effects
            if event.type == pygame.MOUSEMOTION:
                pos = pygame.mouse.get_pos()
                
                # Update button hover states
                if self.game_state == "menu":
                    for button in self.menu_buttons:
                        button.check_hover(pos)
                else:
                    for button in self.game_buttons:
                        button.check_hover(pos)
            
            # Handle mouse clicks
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                pos = pygame.mouse.get_pos()
                
                # Check button clicks
                if self.game_state == "menu":
                    for button in self.menu_buttons:
                        if button.check_hover(pos):
                            result = button.handle_event(event)
                            if result:
                                continue
                elif self.game_state == "comparing":
                    for button in self.game_buttons:
                        if button.check_hover(pos):
                            result = button.handle_event(event)
                            if result:
                                continue
                else:  # playing
                    # Check board clicks
                    if not self.ai_thinking and not self.game.game_over():
                        # Check if click is within board bounds
                        if (BOARD_PADDING <= pos[0] <= BOARD_PADDING + BOARD_SIZE and
                            BOARD_PADDING <= pos[1] <= BOARD_PADDING + BOARD_SIZE):
                            # Convert position to grid coordinates
                            col = (pos[0] - BOARD_PADDING) // CELL_SIZE
                            row = (pos[1] - BOARD_PADDING) // CELL_SIZE
                            
                            # Handle the click
                            self.handle_cell_click(row, col)
                    
                    # Check game buttons
                    for button in self.game_buttons:
                        if button.check_hover(pos):
                            result = button.handle_event(event)
                            if result:
                                continue
        
        return True
    
    def run(self):
        """Main game loop"""
        running = True
        
        while running:
            # Handle events
            running = self.handle_events()
            
            # Process AI move if needed
            if self.game_state == "playing" and self.ai_thinking:
                self.process_ai_move()
            
            # Update animations
            self.animation.update_animations()
            
            # Draw the game
            self.draw(screen)
            
            # Update the display
            pygame.display.flip()
            
            # Control frame rate
            clock.tick(60)
    
    def compare_algorithms(self):
        """Compare the performance of both algorithms"""
        self.game_state = "comparing"
        self.game.reset()
        
        # Setup a non-empty board state for comparison
        self.game.make_move(0, 0, 'X')
        self.game.make_move(1, 1, 'O')
        self.game.current_player = 'X'  # Ensure it's X's turn
        
        # Compare standard Minimax
        std_ai = MinimaxAI('X', use_alpha_beta=False)
        start_time = time.time()
        std_ai.get_best_move(self.game)
        std_time = time.time() - start_time
        std_nodes = std_ai.nodes_evaluated
        
        # Reset and setup the same board state for Alpha-Beta
        self.game.reset()
        self.game.make_move(0, 0, 'X')
        self.game.make_move(1, 1, 'O')
        self.game.current_player = 'X'
        
        # Compare Alpha-Beta Pruning
        ab_ai = MinimaxAI('X', use_alpha_beta=True)
        start_time = time.time()
        ab_ai.get_best_move(self.game)
        ab_time = time.time() - start_time
        ab_nodes = ab_ai.nodes_evaluated
        
        # Store results
        self.performance_stats = {
            "standard": {"nodes": std_nodes, "time": std_time},
            "alpha_beta": {"nodes": ab_nodes, "time": ab_time}
        }
        
        self.game.reset()
        return True

# ... [Rest of the Code (Animation, Button, etc.) Remains Unchanged] ...
class Animation:
    def __init__(self):
        self.animations = []  # List of active animations
        self.winner_anim_progress = 0
    
    def add_piece_animation(self, row, col, piece):
        """Add a new piece animation"""
        self.animations.append({
            'type': 'piece',
            'row': row,
            'col': col,
            'piece': piece,
            'progress': 0
        })
    
    def update_animations(self):
        """Update all animations"""
        # Update piece animations
        for anim in self.animations:
            anim['progress'] += ANIMATION_SPEED
            if anim['progress'] >= 100:
                anim['progress'] = 100
        
        # Remove completed animations
        self.animations = [anim for anim in self.animations if anim['progress'] < 100]
        
        # Update winner animation
        if self.winner_anim_progress < 100:
            self.winner_anim_progress += 2
    
    def draw_animations(self, screen):
        """Draw all active animations"""
        # Draw piece animations
        for anim in self.animations:
            if anim['type'] == 'piece':
                self.draw_piece_animation(screen, anim)
    
    def draw_piece_animation(self, screen, anim):
        """Draw a piece (X or O) animation"""
        progress = anim['progress'] / 100
        row, col = anim['row'], anim['col']
        piece = anim['piece']
        
        # Calculate position
        x = BOARD_PADDING + col * CELL_SIZE + CELL_SIZE // 2
        y = BOARD_PADDING + row * CELL_SIZE + CELL_SIZE // 2
        
        if piece == 'X':
            # Draw X with animation
            line_length = int((CELL_SIZE * 0.8) * progress)
            line_width = int(12 * progress) + 1
            
            start_x1 = x - line_length // 2
            start_y1 = y - line_length // 2
            end_x1 = x + line_length // 2
            end_y1 = y + line_length // 2
            
            start_x2 = x + line_length // 2
            start_y2 = y - line_length // 2
            end_x2 = x - line_length // 2
            end_y2 = y + line_length // 2
            
            pygame.draw.line(screen, X_COLOR, (start_x1, start_y1), (end_x1, end_y1), line_width)
            pygame.draw.line(screen, X_COLOR, (start_x2, start_y2), (end_x2, end_y2), line_width)
            
        else:  # O
            # Draw O with animation
            radius = int((CELL_SIZE * 0.4) * progress)
            line_width = int(10 * progress) + 1
            
            pygame.draw.circle(screen, O_COLOR, (x, y), radius, line_width)
    
    def draw_winning_line(self, screen, winning_line):
        """Draw the winning line with animation"""
        if not winning_line or self.winner_anim_progress == 0:
            return
        
        # Calculate positions
        start_row, start_col = winning_line[0]
        end_row, end_col = winning_line[2]
        
        start_x = BOARD_PADDING + start_col * CELL_SIZE + CELL_SIZE // 2
        start_y = BOARD_PADDING + start_row * CELL_SIZE + CELL_SIZE // 2
        end_x = BOARD_PADDING + end_col * CELL_SIZE + CELL_SIZE // 2
        end_y = BOARD_PADDING + end_row * CELL_SIZE + CELL_SIZE // 2
        
        # Handle diagonal lines
        if start_row != end_row and start_col != end_col:
            # Check which diagonal
            if start_row < end_row and start_col < end_col:  # Top-left to bottom-right
                start_x = BOARD_PADDING + start_col * CELL_SIZE + CELL_SIZE * 0.2
                start_y = BOARD_PADDING + start_row * CELL_SIZE + CELL_SIZE * 0.2
                end_x = BOARD_PADDING + end_col * CELL_SIZE + CELL_SIZE * 0.8
                end_y = BOARD_PADDING + end_row * CELL_SIZE + CELL_SIZE * 0.8
            else:  # Top-right to bottom-left
                start_x = BOARD_PADDING + start_col * CELL_SIZE + CELL_SIZE * 0.8
                start_y = BOARD_PADDING + start_row * CELL_SIZE + CELL_SIZE * 0.2
                end_x = BOARD_PADDING + end_col * CELL_SIZE + CELL_SIZE * 0.2
                end_y = BOARD_PADDING + end_row * CELL_SIZE + CELL_SIZE * 0.8
        
        # Calculate intermediate point based on animation progress
        progress = self.winner_anim_progress / 100
        line_x = start_x + (end_x - start_x) * progress
        line_y = start_y + (end_y - start_y) * progress
        
        # Draw line
        pygame.draw.line(screen, GREEN, (start_x, start_y), (line_x, line_y), 12)
    
    def reset_winner_animation(self):
        """Reset the winner animation"""
        self.winner_anim_progress = 0


class Button:
    def __init__(self, x, y, width, height, text, action=None):
        self.rect = pygame.Rect(x, y, width, height)
        self.text = text
        self.action = action
        self.hovered = False
    
    def draw(self, screen):
        color = BUTTON_HOVER_COLOR if self.hovered else BUTTON_COLOR
        
        # Draw button
        pygame.draw.rect(screen, color, self.rect, border_radius=5)
        pygame.draw.rect(screen, DARK_GRAY, self.rect, 2, border_radius=5)
        
        # Draw text
        text_surface = FONT.render(self.text, True, WHITE)
        text_rect = text_surface.get_rect(center=self.rect.center)
        screen.blit(text_surface, text_rect)
    
    def check_hover(self, pos):
        self.hovered = self.rect.collidepoint(pos)
        return self.hovered
    
    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.hovered and self.action:
                return self.action()
        return None
def main():
    """Main function"""
    gui = GameGUI()
    gui.run()
    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()
