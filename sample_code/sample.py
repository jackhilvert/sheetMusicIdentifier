from PIL import Image
from PIL import ImageFilter

# random number generator
import random

# for drawing text over image
from PIL import ImageDraw

if __name__ == '__main__':
    # load the image
    im = Image.open('first_photograph.png')

    # Check it's  width ,  height, and number of  color channels
    print('Image is %s pixels  wide. ' % im.width)
    print('Image is %s  pixels high. ' % im.height)
    print('Image mode  is %s.' % im.mode)  # (8-bit pixels, black and white)

    # pixels are accessed via a (X,Y) tuple
    print('Pixel value is %s ' % im.getpixel((10, 10)))
    # pixels can be modified by specifying the coordinate and RGB value
    im.putpixel((10, 10), 20)
    print('New pixel value is %s' % im.getpixel((10, 10)))

# Create a new blank color image the same  size as the input
colorim = Image.new('RGB', (im.width, im.height), color=0)
# Loops over the new color image and 
# fills in brighter area that was white first grayscale image we loaded with red colors! 
# Basically we transformed the gray colored first ever photographh into a red colored one!
for x in range(im.width):
    for y in range(im.height):
        grayscale_val = im.getpixel((x, y))
        if grayscale_val >= 190:
            colorim.putpixel((x, y), (grayscale_val, 0, 0))
        else:
            colorim.putpixel((x, y), (grayscale_val, grayscale_val, grayscale_val))

colorim.show()
colorim.save('output.png')

# adding text on top of the image at a random position
colorim_with_text = ImageDraw.Draw(colorim)
# generate a random X-coordinate and a random Y-coordinate
text_x_coord = random.randint(0, colorim.width // 2)
text_y_coord = random.randint(0, colorim.height)
colorim_with_text.text((text_x_coord, text_y_coord),
                       'View from the Window at Le Gras', (255, 255, 255))
colorim.save('output_with_text.png')
