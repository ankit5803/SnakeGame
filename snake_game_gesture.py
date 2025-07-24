import pygame
import random
import sys
import cv2
import numpy as np
import joblib
import mediapipe as mp
import os

# --- Load Gesture Model (Safe) ---
MODEL_DIR = 'models'
try:
    knn = joblib.load(os.path.join(MODEL_DIR, "gesture_knn.pkl"))
    scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
    print("Model and Scaler loaded successfully!")
except Exception as e:
    print("Error loading model or scaler:", e)
    sys.exit()

gesture_labels = {0: 'UP', 1: 'LEFT', 2: 'DOWN', 3: 'RIGHT'}

# --- Mediapipe Setup ---
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

def normalize_landmarks(landmarks):
    landmarks = landmarks.reshape(-1, 2)
    wrist = landmarks[0]
    landmarks -= wrist
    max_val = np.max(np.linalg.norm(landmarks, axis=1))
    if max_val > 0:
        landmarks /= max_val
    return landmarks.flatten()

def get_gesture_direction(hands, frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)
    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        landmarks = np.array([[lm.x, lm.y] for lm in hand_landmarks.landmark]).flatten()
        landmarks = normalize_landmarks(landmarks)
        landmarks = scaler.transform([landmarks])
        pred = knn.predict(landmarks)[0]
        return gesture_labels[pred]
    return None

# --- Pygame Setup ---
pygame.init()
WIDTH, HEIGHT = 800, 600
CELL_SIZE = 20
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
WHITE = (255, 255, 255)

screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Snake Game (Hand Gesture Controlled)")
clock = pygame.time.Clock()
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
    start_x, start_y = WIDTH // 2, HEIGHT // 2
    snake = [(start_x, start_y), (start_x - CELL_SIZE, start_y), (start_x - 2 * CELL_SIZE, start_y)]
    direction = 'RIGHT'
    food_x, food_y = random_food_position()
    score = 0
    speed = 7
    last_direction = direction  # fallback if no gesture detected

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Webcam not found! Exiting.")
        pygame.quit()
        sys.exit()

    with mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7) as hands:
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    cap.release()
                    cv2.destroyAllWindows()
                    pygame.quit()
                    sys.exit()

            ret, frame = cap.read()
            if not ret:
                print("Camera feed lost!")
                break

            frame = cv2.flip(frame, 1)
            gesture_dir = get_gesture_direction(hands, frame)

            # Use gesture or fallback
            if gesture_dir:
                if gesture_dir == 'UP' and last_direction != 'DOWN':
                    last_direction = 'UP'
                elif gesture_dir == 'DOWN' and last_direction != 'UP':
                    last_direction = 'DOWN'
                elif gesture_dir == 'LEFT' and last_direction != 'RIGHT':
                    last_direction = 'LEFT'
                elif gesture_dir == 'RIGHT' and last_direction != 'LEFT':
                    last_direction = 'RIGHT'

            # Move snake
            snake = move_snake(snake, last_direction)

            # Check food
            if snake[0][0] == food_x and snake[0][1] == food_y:
                snake = grow_snake(snake)
                food_x, food_y = random_food_position()
                score += 1

            # Check collision
            if check_collision(snake):
                cap.release()
                cv2.destroyAllWindows()
                return score

            # Draw game
            screen.fill(BLACK)
            draw_snake(snake)
            draw_food(food_x, food_y)
            draw_score(score)
            pygame.display.flip()

            # Show webcam feed (optional)
            cv2.imshow("Hand Tracking (ESC to quit)", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                cap.release()
                cv2.destroyAllWindows()
                pygame.quit()
                sys.exit()

            clock.tick(speed)

def main():
    while True:
        final_score = game_loop()

        # Game Over screen
        screen.fill(BLACK)
        over_text = font.render(f"Game Over! Score: {final_score}", True, RED)
        restart_text = font.render("Press R to Restart or Q to Quit", True, WHITE)
        screen.blit(over_text, (WIDTH // 2 - 150, HEIGHT // 2 - 40))
        screen.blit(restart_text, (WIDTH // 2 - 200, HEIGHT // 2 + 10))
        pygame.display.flip()

        waiting = True
        while waiting:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_r:
                        waiting = False
                    elif event.key == pygame.K_q:
                        pygame.quit()
                        sys.exit()

if __name__ == "__main__":
    main()
