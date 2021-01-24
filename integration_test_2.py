import os
from PIL import Image
from pic2sgf import Pic2Array

interpreter = Pic2Array(4)

fn = "pic2sgf/testing/fotos-desde-reddit/r (16).jpg"
img = Image.open(fn)
try:
    board, report = interpreter(img)
except Exception as e:
    print(e)
else:
    print(report)
