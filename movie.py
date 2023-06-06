import os
import imageio
import re

# assign directory
directory = 'dshift_5min_dist_plots_unfiltered'

filenames = []
# iterate over files in that directory
for filename in os.scandir(directory):
    if filename.is_file():
        filenames.append('./' + directory + '/' + filename.name)

filenames.sort(key=lambda f: int(re.sub('\D', '', f)))

frames = []
for t in range(0, 275):
    image = imageio.v2.imread(filenames[t])
    frames.append(image)

imageio.mimsave('./example.gif',    # output gif
                frames,             # array of input frames
                fps=10)              # optional: frames per second

