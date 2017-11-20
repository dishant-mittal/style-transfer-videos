import pylab
import imageio
#comment the following line if ffmpeg is not installed
#imageio.plugins.ffmpeg.download()
filename = './v2.mp4'
vid = imageio.get_reader(filename,  'ffmpeg')
num_frames=vid._meta['nframes']
print num_frames
#nums = [10, 287]
for num in range(1, num_frames):
    image = vid.get_data(num)
    #print type(image)
    imageio.imwrite('./data/frame' +str(num) + '.png', image[:, :, :])
    #fig = pylab.figure()
    #fig.suptitle('image #{}'.format(num), fontsize=20)
    #pylab.imshow(image)
#pylab.show()