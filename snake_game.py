import pygame
import random
import sys

# Initialize pygame
pygame.init()

# Screen dimensions
WIDTH, HEIGHT = 800, 600
CELL_SIZE = 20

# Colors
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
WHITE = (255, 255, 255)

# Set up screen
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Snake Game")

# Clock for controlling speed
clock = pygame.time.Clock()

# Font for score display
font = pygame.font.SysFont("Arial", 28)

def random_food_position():
    x = random.randrange(0, WIDTH // CELL_SIZE) * CELL_SIZE
    y = random.randrange(0, HEIGHT // CELL_SIZE) * CELL_SIZE
    return x, y

def move_snake(snake, direction):
    x, y = snake[0]
    if direction == 'UP':
        y -= CELL_SIZE
    elif direction == 'DOWN':
        y += CELL_SIZE
    elif direction == 'LEFT':
        x -= CELL_SIZE
    elif direction == 'RIGHT':
        x += CELL_SIZE
    new_head = (x, y)
    snake = [new_head] + snake[:-1]
    return snake

def check_collision(snake):
    head = snake[0]
    if head[0] < 0 or head[0] >= WIDTH or head[1] < 0 or head[1] >= HEIGHT:
        return True
    if head in snake[1:]:
        return True
    return False

def draw_snake(snake):
    for x, y in snake:
        pygame.draw.rect(screen, GREEN, (x, y, CELL_SIZE, CELL_SIZE))

def draw_food(x, y):
    pygame.draw.rect(screen, RED, (x, y, CELL_SIZE, CELL_SIZE))

def grow_snake(snake):
    snake.append(snake[-1])
    return snake

def draw_score(score):
    text = font.render(f"Score: {score}", True, WHITE)
    screen.blit(text, (10, 10))

def game_loop():
    # Initialize snake and food
    start_x = WIDTH // 2
    start_y = HEIGHT // 2
    snake = [(start_x, start_y), (start_x - CELL_SIZE, start_y), (start_x - 2 * CELL_SIZE, start_y)]
    direction = 'RIGHT'
    speed = 5
    food_x, food_y = random_food_position()
    score = 0

    running = True
    while running:
        # Event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP and direction != 'DOWN':
                    direction = 'UP'
                elif event.key == pygame.K_DOWN and direction != 'UP':
                    direction = 'DOWN'
                elif event.key == pygame.K_LEFT and direction != 'RIGHT':
                    direction = 'LEFT'
                elif event.key == pygame.K_RIGHT and direction != 'LEFT':
                    direction = 'RIGHT'

        # Move snake
        snake = move_snake(snake, direction)

        # Check for food collision
        if snake[0][0] == food_x and snake[0][1] == food_y:
            snake = grow_snake(snake)
            food_x, food_y = random_food_position()
            score += 1

        # Check for game over
        if check_collision(snake):
            return score  # return final score to handle restart

        # Draw everything
        screen.fill(BLACK)
        draw_snake(snake)
        draw_food(food_x, food_y)
        draw_score(score)
        pygame.display.flip()

        # Control speed
        clock.tick(speed)

def main():
    while True:
        final_score = game_loop()

        # Show Game Over screen
        screen.fill(BLACK)
        game_over_text = font.render(f"Game Over! Score: {final_score}", True, RED)
        restart_text = font.render("Press R to Restart or Q to Quit", True, WHITE)
        screen.blit(game_over_text, (WIDTH // 2 - 150, HEIGHT // 2 - 40))
        screen.blit(restart_text, (WIDTH // 2 - 200, HEIGHT // 2 + 10))
        pygame.display.flip()

        # Wait for restart or quit
        waiting = True
        while waiting:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_r:
                        waiting = False  # restart
                    elif event.key == pygame.K_q:
                        pygame.quit()
                        sys.exit()

if __name__ == "__main__":
    main()
