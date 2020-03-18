# pic2sgf

A python package to read Go board positions from pictures using deep learning.

## Installation

Install pytorch, torchvision, numpy, scipy and PIL

## Usage

Create a Pic2Array object, load an image with PIL, and simple use the __call__ method:

```python
pic2array = Pic2Array(board_size = 9)
image = Image.open('<path_to_image>.jpg').resize((512, 384))
board, is_ok, _ = pic2array(image)
```

The variable board is an array with -1 for black, 1 for white and zero for empty places. The variable
is_ok is a boolean indicating if the image could be interpreted.

See test.py for an example.
