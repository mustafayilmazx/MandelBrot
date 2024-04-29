import numpy as np
from PIL import Image
from concurrent.futures import ThreadPoolExecutor

def mandelbrot(c, max_iter):
    z = 0
    n = 0
    while abs(z) <= 2 and n < max_iter:
        z = z*z + c
        n += 1
    if n == max_iter:
        return 255  # Considered a member of the Mandelbrot set
    else:
        return n % 256  # Escape time modulo 256 for color

def compute_row(y, xmin, xmax, width, max_iter):
    row = np.linspace(xmin, xmax, width)
    return [255 - mandelbrot(complex(x, y), max_iter) for x in row]

def create_mandelbrot_image(xmin, xmax, ymin, ymax, width, height, max_iter, filename):
    image = np.zeros((height, width), dtype=np.uint8)
    y_values = np.linspace(ymin, ymax, height)
    with ThreadPoolExecutor() as executor:
        results = executor.map(compute_row, y_values, [xmin]*height, [xmax]*height, [width]*height, [max_iter]*height)
    for i, row in enumerate(results):
        image[i] = row
    img = Image.fromarray(image, mode="L")
    img.save(filename)

# Parameters
xmin, xmax, ymin, ymax = -2.0, 1.0, -1.5, 1.5
img_width, img_height = 4000, 3000
max_iter = 1023  # Maximum iterations increased to 1023
filename = "output_multithreaded.png"

# Create and save the image
create_mandelbrot_image(xmin, xmax, ymin, ymax, img_width, img_height, max_iter, filename)
