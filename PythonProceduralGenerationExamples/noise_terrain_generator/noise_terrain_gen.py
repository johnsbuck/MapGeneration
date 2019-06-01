import argparse
import random
import json
import os

import pygame

from NoiseGenerator import PerlinNoise, ValueNoise, SimplexNoise


def get_name(main_title, folder, sep="_"):
    count = 0
    if not os.path.isdir(folder):
        os.mkdir(folder)
    while os.path.isfile(folder + "/" + main_title + sep + str(count) + ".bmp"):
        count += 1
    return main_title + sep + str(count)


def noise_terrain_gen(data=None):
    if data is None:
        # ------------------------------------------------
        # Define Default Parameters
        # ------------------------------------------------

        # Define Land Colors
        default_land_colors = [(0.15, [20, 75, 190]),  # Deep Ocean
                               (0.32, [29, 100, 210]),  # Ocean
                               (0.52, [38, 110, 225]),  # Shallow
                               (0.60, [47, 128, 235]),  # Coast
                               (0.64, [254, 251, 192]),  # Beach
                               (0.78, [60, 204, 62]),  # Grass
                               (0.88, [80, 114, 16]),  # Forest
                               (0.95, [91, 46, 39]),  # Mountain
                               (0.98, [180, 92, 80]),  # Snowy Mountain
                               (1.00, [255, 255, 255])  # Snow
                               ]

        # Define Cloud Colors (w/ Alpha)
        # Kept commented colors for more detailed cloud coloring example
        default_cloud_colors = [(0.65, [255, 255, 255, 0]),     # No Clouds
                                # (0.70, [255, 255, 255, 30]),
                                (0.75, [255, 255, 255, 60]),
                                # (0.80, [255, 255, 255, 90]),
                                (0.85, [255, 255, 255, 120]),
                                # (0.90, [255, 255, 255, 150]),
                                (1.00, [255, 255, 255, 180]),   # originally 0.95
                                # (1.00, [255, 255, 255, 210]), # Maximum Clouds
                                ]

        # Define Default Parameter Dictionary
        data = {
            "land": {
                "seed": random.randrange(0, 2 ** 32 - 1),
                "frequency": 4,
                "octaves": 3,
                "lacunarity": 1,
                "persistence": 0.3,
                "colors": default_land_colors,
                "noise": {
                    "type": "perlin",
                    "dim": 2,
                },
            },
            "cloud": {
                "seed": random.randrange(0, 2 ** 32 - 1),
                "frequency": 4,
                "octaves": 3,
                "lacunarity": 1,
                "persistence": 0.3,
                "colors": default_cloud_colors,
                "noise": {
                    "type": "perlin",
                    "dim": 2
                },
            },
            "resolution": 256,
            "scale": 2,
            "folder": "gen_terrain",
            "name": get_name("terrain_gen", "gen_terrain"),
            "pygame_wait": 0,
        }

    # ------------------------------------------------
    # Generate Noise
    # ------------------------------------------------

    print("BEGIN GENERATION")

    noises = {"perlin": PerlinNoise, "value": ValueNoise, "simplex": SimplexNoise}

    # Define Noise Generator with Seed
    if data["land"]["noise"]["type"].lower() in noises:
        land_noise = noises[data["land"]["noise"]["type"]](data["land"]["seed"])
        land_method = land_noise.NOISE_LIST[data["cloud"]["noise"]["dim"] - 1]
    else:
        raise ValueError("Land Noise Type given is not valid")

    if data["cloud"]["noise"]["type"].lower() in noises:
        cloud_noise = noises[data["cloud"]["noise"]["type"]](data["cloud"]["seed"])
        cloud_method = cloud_noise.NOISE_LIST[data["cloud"]["noise"]["dim"] - 1]
    else:
        raise ValueError("Cloud Noise Type given is not valid")

    # Generate Noise
    print("GENERATE NOISE")
    land_noise = land_noise(data["resolution"], land_method, data["land"]["frequency"],
                            data["land"]["octaves"], data["land"]["lacunarity"],
                            data["land"]["persistence"])
    cloud_noise = cloud_noise(data["resolution"], cloud_method, data["cloud"]["frequency"],
                              data["cloud"]["octaves"], data["cloud"]["lacunarity"],
                              data["cloud"]["persistence"])

    # Feature Scale to [0, 1]
    print(land_noise.shape)
    print(land_noise.min(), land_noise.max())
    land_noise = (land_noise - land_noise.min()) / (land_noise.max() - land_noise.min())
    cloud_noise = (cloud_noise - cloud_noise.min()) / (cloud_noise.max() - cloud_noise.min())

    # ------------------------------------------------
    # Generate Bitmap
    # ------------------------------------------------

    print("GENERATE BITMAP")

    # Generate Pygame Screen
    xSize = data["resolution"] * data["scale"]
    ySize = data["resolution"] * data["scale"]
    screen = pygame.display.set_mode((xSize, ySize))
    screen.fill((0, 0, 0))

    # Generate Land and Cloud Layers
    land_screen = pygame.Surface(screen.get_size(), pygame.SRCALPHA)
    cloud_screen = pygame.Surface(screen.get_size(), pygame.SRCALPHA)
    pygame.display.set_caption("Noisy Terrain Creation")

    # Begin Generating Terrain on Pygame
    pygame.init()
    for x in range(xSize // data["scale"]):
        x *= data["scale"]
        for y in range(ySize // data["scale"]):
            y *= data["scale"]
            # Set each space to a defined land color
            for i in range(len(data["land"]["colors"])):
                if land_noise[x // data["scale"], y // data["scale"]] <= data["land"]["colors"][i][0]:
                    pygame.draw.rect(land_screen, data["land"]["colors"][i][1],
                                     (x, y, data["scale"], data["scale"]))
                    break

            # Set each space to a defined cloud color
            for i in range(len(data["cloud"]["colors"])):
                if cloud_noise[x // data["scale"], y // data["scale"]] <= data["cloud"]["colors"][i][0]:
                    pygame.draw.rect(cloud_screen, data["cloud"]["colors"][i][1],
                                     (x, y, data["scale"], data["scale"]))
                    break

    # Insert Land and Cloud layers onto Pygame Screen
    screen.blit(land_screen, (0, 0))
    screen.blit(cloud_screen, (0, 0))
    pygame.display.flip()
    pygame.time.wait(data["pygame_wait"])

    print("FINISHED GENERATION")

    # Save Terrain as Bitmap
    print(data["folder"] + "/" + data["name"])
    pygame.image.save(screen, data["folder"] + "/" + data["name"] + ".bmp")

    # Save Parameters as JSON
    with open(data["folder"] + "/" + data["name"] + ".json", "w+") as f:
        json.dump(data, f)

    print("FINISHED SAVING")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Perlin noise_terrain_generator Parameter Parser")
    parser.add_argument("json", type=str, nargs="?", default=None,
                        help="An existing JSON file with proper structure defined in README.md")
    args = parser.parse_args()
    user_data = None
    if args.json is not None:
        with open(args.json, "r") as json_file:
            user_data = json.load(json_file)
    noise_terrain_gen(user_data)

    print("END OF PROGRAM")
