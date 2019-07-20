import os, sys
import os.path
import numpy as np

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

def dataloader(csv_path="filelist.csv"):
    filenames = open(csv_path).readlines()

    frame_all = [filename.split(',')[0].strip() for filename in filenames]
    nframes = [int(filename.split(',')[1].strip()) for filename in filenames]

    all_index = np.arange(len(nframes))
    np.random.shuffle(all_index)

    refs_train = []

    for index in all_index:
        ref_num = 3
        frame_interval = 4  #

        # compute frame index (ensures length(image set) >= random_interval)
        refs_images =[]

        n_frames = nframes[index]
        frame_indices = np.arange(1, n_frames, frame_interval)  # start from 1
        total_batch, batch_mod = divmod(len(frame_indices), ref_num)
        if batch_mod > 0:
            frame_indices = frame_indices[:-batch_mod]
        frame_indices_batches = np.split(frame_indices, total_batch)
        for batches in frame_indices_batches:
            ref_images = [os.path.join(frame_all[index], 'image_{:05d}.jpg'.format(frame))
                          for frame in batches]
            refs_images.append(ref_images)

        refs_train.extend(refs_images)

    return refs_train

if __name__ == '__main__':
    x = dataloader()
