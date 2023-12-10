
import imageio
import glob
import PIL
# Display a single image using the epoch number
def display_image(epoch_no):
    return PIL.Image.open('image_at_epoch_{:04d}.jpg'.format(epoch_no))


anim_file = 'flowergan.gif'

with imageio.get_writer(anim_file, mode='I') as writer:
    filenames = glob.glob('./pic_1/image*.jpg')
    filenames = sorted(filenames)
    for filename in filenames:
        image = imageio.imread(filename)
        writer.append_data(image)
    image = imageio.imread(filename)
    writer.append_data(image)

