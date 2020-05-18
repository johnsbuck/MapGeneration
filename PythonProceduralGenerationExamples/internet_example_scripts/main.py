import pygame
import numpy as np
import random
import time

# Settings
size = 80  # count radius, controls biome size
pixel = 2  # controls resolution
k = 1.0  # relative altitude (0.6 < x < 1.5, for best results)
grass = .7  # color level
rock = .9  # ''    ''
snow = .95  # ''    ''

dotsx = []  # contains all x-coordinates for the dots
dotsy = []  # contains all y-coordinates as well as the weighting


def generate_noise(width, height):
    noise_map = []
    # Populate a noise map with 0s
    for y in range(height):
        new_row = []
        for x in range(width):
            new_row.append(0)
        noise_map.append(new_row)

    # Progressively apply variation to the noise map but changing values + or -
    # 5 from the previous entry in the same list, or the average of the
    # previous entry and the entry directly above
    new_value = 0
    top_of_range = 0
    bottom_of_range = 0
    for y in range(height):
        for x in range(width):
            if x == 0 and y == 0:
                continue
            if y == 0:  # If the current position is in the first row
                new_value = noise_map[y][x - 1] + random.randint(-1000, +1000)
            elif x == 0:  # If the current position is in the first column
                new_value = noise_map[y - 1][x] + random.randint(-1000, +1000)
            else:
                minimum = min(noise_map[y][x - 1], noise_map[y-1][x])
                maximum = max(noise_map[y][x - 1], noise_map[y-1][x])
                average_value = minimum + ((maximum-minimum)/2.0)
                new_value = average_value + random.randint(-1000, +1000)
            noise_map[y][x] = new_value
            # check whether value of current position is new top or bottom
            # of range
            if new_value < bottom_of_range:
                bottom_of_range = new_value
            elif new_value > top_of_range:
                top_of_range = new_value
    # Normalises the range, making minimum = 0 and maximum = 1
    difference = float(top_of_range - bottom_of_range)
    for y in range(height):
        for x in range(width):
            noise_map[y][x] = (noise_map[y][x] - bottom_of_range)/difference
    return noise_map


xSize, ySize = 640, 480
screen = pygame.display.set_mode((xSize, ySize))
pygame.display.set_caption("Noise Creation")

noise = generate_noise(xSize // pixel, ySize // pixel)
noise = np.array(noise)
noise = noise.transpose()

pygame.init()

# Create Landscape
# screen.fill([29, 109, 210])

# bitmap = np.zeros((xSize, ySize, 3))
for x in range(xSize // pixel):
    x *= pixel
    for y in range(ySize // pixel):
        y *= pixel
        try:
            if noise[x//pixel][y//pixel] > snow:
                pygame.draw.rect(screen, [255, 255, 255], (x, y, pixel, pixel))
            elif noise[x//pixel][y//pixel] > rock:
                pygame.draw.rect(screen, [91, 46, 39], (x, y, pixel, pixel))
            elif noise[x//pixel][y//pixel] > grass:
                pygame.draw.rect(screen, [60, 204, 62], (x, y, pixel, pixel))
            else:
                pygame.draw.rect(screen, [29, 109, 210], (x, y, pixel, pixel))
        except IndexError:
            raise IndexError("List out of range: " + str(x // pixel) + ", " + str(y // pixel))

        pygame.display.flip()

pygame.image.save(screen, "keep.bmp")

print("DONE")
time.sleep(3)
