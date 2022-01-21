from environment import Grid
from PIL import Image
import cv2
import numpy as np

def display_ui(done, environment):
    env = environment
    array_for_showing = np.zeros((env.env_size, env.env_size, 3), dtype=np.uint8)
    for row in range(0, env.env_size):
        for col in range(0, env.env_size):
            if env.grid[row][col] == Grid.AGENT:
                array_for_showing[row][col] = (255, 175, 0)
            elif env.grid[row][col] == Grid.BOMB:
                array_for_showing[row][col] = (0, 0, 255)
            elif env.grid[row][col] == Grid.GOLD:
                array_for_showing[row][col] = (50, 224, 224)
            else:
                array_for_showing[row][col] = (0, 0, 0)

    img = Image.fromarray(array_for_showing, "RGB")
    img = img.resize((300, 300), resample=Image.BOX)
    cv2.imshow("Grid Game", np.array(img))
    if done:
        cv2.waitKey(320)
    else:
        cv2.waitKey(300)
