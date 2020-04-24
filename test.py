from pic2sgf import Pic2Array
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize
from torchvision.datasets.folder import default_loader
from tqdm import tqdm
from PIL import Image

pic2array = Pic2Array(1)

plt.figure(figsize=(20, 16))
for i in tqdm(range(12)):
    image = default_loader(f'pic2sgf/test_images/{i+1}.jpg')

    configuration, report = pic2array(image)

    plt.subplot(4, 6, 2*i+1)
    plt.imshow(image)
    plt.subplot(4, 6, 2*i+2)
    plt.imshow(report['board_image'], extent=(0,1,0,1))
    plt.imshow(configuration, extent=(0,1,0,1), alpha=0.25, cmap='bwr', norm = Normalize(vmin = -1, vmax = 1))

print(report)

plt.show()