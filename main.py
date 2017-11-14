import pygame

# Define colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GREY = (200,200,200)
GREEN = (0, 255, 0)
RED = (255, 0, 0)

# Set cell Size
WIDTH = 10
HEIGHT = 10

# Set Margin
MARGIN = 1

# Create Grid
grid = []
for row in range(60):
    # Add an empty array that will hold each cell
    # in this row
    grid.append([])
    for column in range(60):
        grid[row].append(0)  # Append a cell

# grid[1][5] = 1

# Initialize
pygame.init()

# Set window H and W
WINDOW_SIZE = [660, 660]
screen = pygame.display.set_mode(WINDOW_SIZE)

# Set window title
pygame.display.set_caption("Play-emergence")

# Loop until the user clicks the close button.
done = False

# Used to manage how fast the screen updates
clock = pygame.time.Clock()

while not done:
    for event in pygame.event.get():  # Got event
        if event.type == pygame.QUIT:  # On close
            done = True  # Let program go
        # Debug Mouse functionality    
        # elif event.type == pygame.MOUSEBUTTONDOWN:
        #     pos = pygame.mouse.get_pos()
        #     # Change the x/y screen coordinates to grid coordinates
        #     column = pos[0] // (WIDTH + MARGIN)
        #     row = pos[1] // (HEIGHT + MARGIN)
        #     # Set that location to one
        #     grid[row][column] = 1
        #     print("Click ", pos, "Grid coordinates: ", row, column)

    # Set the screen background
    screen.fill(GREY)

    # Draw the grid
    for row in range(60):
        for column in range(60):
            color = WHITE
            if grid[row][column] == 1:
                color = GREEN
            pygame.draw.rect(screen,
                             color,
                             [(MARGIN + WIDTH) * column + MARGIN,
                              (MARGIN + HEIGHT) * row + MARGIN,
                              WIDTH,
                              HEIGHT])

    # Limit to 60 frames per second
    clock.tick(60)

    # Go ahead and update the screen with what we've drawn.
    pygame.display.flip()

# Be IDLE friendly. If you forget this line, the program will 'hang'
# on exit.
pygame.quit()
