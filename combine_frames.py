# import cv2
# import argparse
# import os
# import re

# # Construct the argument parser and parse the arguments
# ap = argparse.ArgumentParser()
# ap.add_argument("-ext", "--extension", required=False, default='png', help="extension name. default is 'png'.")
# ap.add_argument("-o", "--output", required=False, default='output.mp4', help="output video file")
# args = vars(ap.parse_args())

# # Arguments
# dir_path = './data'
# ext = args['extension']
# output = args['output']

# images = []
# for f in os.listdir(dir_path):
#     if f.endswith(ext):
#         images.append(f)


# def name2num(name):
#    m = re.search('frame(\d+)\.?.*', name)
#    return int(m.group(1),10)



# images.sort(key=name2num)

# # Determine the width and height from the first image
# image_path = os.path.join(dir_path, images[0])
# frame = cv2.imread(image_path)
# cv2.imshow('video',frame)
# height, width, channels = frame.shape

# # Define the codec and create VideoWriter object
# fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Be sure to use lower case
# out = cv2.VideoWriter(output,cv2.VideoWriter_fourcc('M','J','P','G'), 3, (width, height))

# for image in images:

#     image_path = os.path.join(dir_path, image)
#     frame = cv2.imread(image_path)

#     out.write(frame) # Write out frame to video

#     cv2.imshow('video',frame)
#     if (cv2.waitKey(1) & 0xFF) == ord('q'): # Hit `q` to exit
#         break

# # Release everything if job is finished
# out.release()
# cv2.destroyAllWindows()


# print("The output video is {}".format(output))

import imageio
images = []

#for filename in filenames:
for x in range(1,467):
    images.append(imageio.imread('./data/frame'+str(x)+'.png'))

imageio.mimsave('./output.mp4', images)