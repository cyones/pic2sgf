import os
from PIL import Image
from pic2array import Pic2Array
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize

interpreter = Pic2Array(4)

fn = "./testing/fotos-desde-reddit/r (16).jpg"
img = Image.open(fn)
fig = plt.figure()
plt.subplot(1, 2, 1)
plt.imshow(img)
try:
    board, report = interpreter(img)
    print(report)
except Exception as e:
    plt.suptitle(f"Failed: {e}")
    print(e)
else:
    plt.suptitle("Ok")
    plt.subplot(1, 2, 2)
    plt.imshow(report['board_image'], extent=(0,1,0,1))
    plt.imshow(board,
        norm = Normalize(vmin = -1, vmax = 1),
        cmap='bwr',
        extent=(0,1,0,1),
        alpha=0.25)
finally:
    plt.show()
