from pic2sgf import Pic2Array
from PIL import Image
from matplotlib import pyplot as plt

pic2array = Pic2Array()

plt.figure(figsize=(12, 8))
for i in range(6):
    image = Image.open(f'pic2sgf/test_images/{i+1}.jpg').resize((512, 384))
    board, board_image = pic2array(image)
    
    plt.subplot(3, 4, 2*i+1)
    plt.imshow(image)
    plt.subplot(3, 4, 2*i+2)
    plt.imshow(board_image, extent=(0,1,0,1))
    plt.imshow(board, extent=(0,1,0,1), alpha=0.5, cmap='bwr')
    plt.title(f"Board size: {board.shape}")

plt.show()

