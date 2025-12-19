from pathlib import Path

import h5py
import imageio.v2 as imageio
import matplotlib
import numpy as np

from quickpde.initial_conditions import gaussian_random_field

files = list(Path('data/pyqg').rglob('*.h5'))
for i, f in enumerate(files):
  x = np.array(h5py.File(f)['data'])
  x += np.abs(np.min(x))
  x /= np.max(x)

  cmap = matplotlib.colormaps['viridis']

  y = (255 * np.array(cmap(x))).astype('uint8')
  vid_writer = imageio.get_writer(
      f'test_{i}' + '.mp4',
      fps=60,
      quality=10,  # for some codecs
      ffmpeg_params=[
          "-preset",
          "slow",
          "-crf",
          "1"  # lower = higher quality, 15–18 is visually lossless
      ],
      codec="libx264",
      bitrate="30M",
  )
  for t in x:
    n_bins = 256
    rgb = (255 * cmap(np.array(t))[..., :3]).astype(np.uint8)
    vid_writer.append_data(rgb)
  vid_writer.close()
