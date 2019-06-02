# Noise Terrain Generator

![Terrain Example](gen_terrain/terrain_gen_3.bmp)

This generator uses a Perlin-based noise generator to create values per pixel that correspond 
to a specific terrain. This is then saved as a bitmap (.bmp) with the JSON file being saved 
with the same name.

## JSON Format

The JSON format itself is fairly convoluted at this point, and is suggested that if you wish to
make your own changes, to use one of the examples in the **gen_terrain folder** as a **template**.

~~~
{
    "land": {
        "seed": An integer from 0 to (2^32 - 1) (inclusive). Defines the seed used to randomize the noise,
        "frequency": A float that is greater than 0. Used in noise functions, 
        "octaves": An integer from 1 to 16 (inclusive). Used in noise functions, 
        "lacunarity": A float that is greater than 1. Used in noise functions, 
        "persistence": A float from 0 to 1 (inclusive). Used in noise functions, 
        "colors": An array of arrays containing a percentage float from 0 to 1 (inclusive) and an array of
                  3 to 4 elements of values from 0 to 255 (inclusive) representing either RGB or
                  RGBA. Should be arranged from lowest percentage to highest and the last value
                  should be 1.0. Used to color each like a heightmap.
                  
                  Example:
                   [[0.15, [20, 75, 190]], 
                   [0.32, [29, 100, 210]], 
                   [0.52, [38, 110, 225]], 
                   [0.6, [47, 128, 235]], 
                   [0.64, [254, 251, 192]], 
                   [0.78, [60, 204, 62]], 
                   [0.88, [80, 114, 16]], 
                   [0.95, [91, 46, 39]], 
                   [0.98, [180, 92, 80]], 
                   [1, [255, 255, 255]]], 
        "noise": {
            "type": Either the string "perlin", "value", or "simplex". Used to choose a specific noise generator, 
            "dim": An integer from 1 to 3 (inclusive) or 1 to 2 (inclusive) if type is simplex. 
                   Used to choose the dimension of the noise.
            }
    }, 
    "cloud": {
        "seed": An integer from 0 to (2^32 - 1) (inclusive). Defines the seed used to randomize the noise,
        "frequency": A float that is greater than 0. Used in noise functions, 
        "octaves": An integer from 1 to 16 (inclusive). Used in noise functions, 
        "lacunarity": A float that is greater than 1. Used in noise functions, 
        "persistence": A float from 0 to 1 (inclusive). Used in noise functions, 
        "colors": An array of arrays containing a percentage float from 0 to 1 (inclusive) and an array of
                  3 to 4 elements of values from 0 to 255 (inclusive) representing either RGB or
                  RGBA. Should be arranged from lowest percentage to highest and the last value
                  should be 1.0. Used to color each like a heightmap.
                  
                  Example:
                    [[0.65, [255, 255, 255, 0]], 
                    [0.75, [255, 255, 255, 60]], 
                    [0.85, [255, 255, 255, 120]], 
                    [1.0, [255, 255, 255, 180]]], 
        "noise": {
            "type": Either the string "perlin", "value", or "simplex". Used to choose a specific noise generator, 
            "dim": An integer from 1 to 3 (inclusive) or 1 to 2 (inclusive) if type is simplex. 
                   Used to choose the dimension of the noise.
            }
    }, 
    "resolution": An integer that is greater than 3. Used to define amount of noise (resolution * resolution) 
                  and size of final result, 
    "scale": An integer that is greater than 0. Used to scale final result size (resolution * scale, resolution * scale), 
    "folder": A string with a valid foldername (no "/" or "\"). Saved in specified folder, 
    "name": A string with a valid filename (no typing such as ".bmp"). Generates JSON and BMP files with the same filename, 
    "pygame_wait": An non-negative integer. Amount of time (in msec) to be displayed on screen after generation.
}
~~~
