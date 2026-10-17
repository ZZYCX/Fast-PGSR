import numpy as np
import matplotlib.pyplot as plt
import random
import OpenGL.GL as gl
from PIL import Image
import datetime

def capture_screenshot(window_width, window_height):
    gl.glFlush()
    gl.glFinish()

    gl.glReadBuffer(gl.GL_BACK)
    data = gl.glReadPixels(0, 0, window_width, window_height, gl.GL_RGB, gl.GL_UNSIGNED_BYTE)

    image = np.frombuffer(data, dtype=np.uint8).reshape(window_height, window_width, 3)

    image = np.flipud(image)

    img = Image.fromarray(image)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"screenshot_{timestamp}.png"

    img.save(filename)

def vis_temperature_field(temperature_field: np.ndarray):
    cmap = plt.get_cmap("inferno") 
    temp_normalized = (temperature_field - temperature_field.min()) / (temperature_field.max() - temperature_field.min())
    colors = cmap(temp_normalized)[:, :3]
    return colors

def generate_non_gold_color():
    """Generate a random color that avoids golden tones"""
    while True:
        r, g, b = random.random(), random.random(), random.random()
        if not (r > 0.7 and g > 0.7 and b < 0.4):
            if not (r > 0.8 and g > 0.5 and b < 0.3):
                return [r, g, b]