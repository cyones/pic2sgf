from pic2array import Pic2Array
from PIL import Image
from matplotlib import pyplot as plt

pic2array = Pic2Array(board_size = 9)

plt.figure(figsize=(12, 8))
for i in range(4):
    image = Image.open(f'test_images/{i+1}.jpg').resize((512, 384))
    board, is_ok, board_image = pic2array(image)
    
    plt.subplot(2, 4, 2*i+1)
    plt.imshow(image)
    plt.subplot(2, 4, 2*i+2)
    plt.imshow(board_image, extent=(0,1,0,1))
    plt.imshow(board, extent=(0,1,0,1), alpha=0.25, cmap='bwr')
    plt.title('Correct' if is_ok else 'Incorrect')

plt.show()

