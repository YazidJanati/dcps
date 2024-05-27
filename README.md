# Divide-and-Conquer Posterior Sampling for Denoising Diffusion Priors

The code of DCPS algorithm for solving Bayesian inverse problems with Diffusion models as prior.
The algorithm solves linear and non-linear problems with either Gaussian or Poisson noise.


## Code installation

Install the code in editable mode

```bash
pip install -e .
```

This command will also download the code dependencies.
Further details about the code can be found in ``pyproject.toml``.

For convenience, the code of these repositories were moved inside ``src`` folder

- https://github.com/gabrielvc/mcg_diff
- https://github.com/bahjat-kawar/ddrm
- https://github.com/openai/guided-diffusion
- https://github.com/NVlabs/RED-diff
- https://github.com/mlomnitz/DiffJPEG

to avoid installation conflicts.


## Large files

The models checkpoints, datasets were ignored as they contain large files.
Make sure to create a folder ``large_files`` and download the right files and folders.

To avoid path conflict, ensure to insert in ``src/local_paths.py`` script

- the absolute path of the repository
- the path of the folder ``large_files``

and update the ``model_path`` in the configuration files ``ffhq_model.yaml`` and ``imagenet_model.yaml``.

The ``large_files`` folder have the following structure.
Make sure to preserve it.

```
  large_files/
  ├── ddm-inv-problems/
  ├──── ffhq/
  |    └── validation_set/
  |       └── im1.png
  |       └── ...
  |    └── ffhq_mt.pt
  ├──── imagenet/
  |    └── validation_set/
  |       └── im1.png
  |       └── ...
  ├──── masks_img256/
  |    └── inpainting_middle.pt
  |    └── ...
  |—— trajectories/
  |    └── raw_data/
  |       └── ucy/
  |          └── students_1.txt
  |          └── students_3.txt
  |    └── checkpoints/
  |       └── ucy_len_20_n_diff_steps_1000.pt
```


## A tour on the repository scripts

Both ``demo_images.py`` and ``demo_trajectory.py`` scripts can be used to solve inverse problems with an algorithm among the considered ones.

Use the the dataclass ``Config`` to customize the behavior of the script, more precisely

#### demo_images.py

- **model** (``str``) : Either "celebhq", "ffhq", or "imagenet"
- **n_steps** (``int``) : The number of diffusion steps to use
- **algo** (``str``) : Either "dcps", "ddrm", "dps", "mcgdiff", "pgdm", or "reddiff"
- **img_idx** (``str``) : the relative path of the image
- **task** (``str``) : Either "outpainting_half", "inpainting_middle", "outpainting_expand", "sr4", "sr16", "jpeg{QUALITY}" where QUALITY is an int between 1 and 100, for example "jpeg8"
- **noise_type** (``str``) : Either "gaussian" or "poisson"
- **std** (``float`` strictly positive) : the standard deviation of the noise
- **poisson_rate** (``float`` in (0, 1)) : rate of poisson noise
- **n_samples** (``int``): the number of samples to generate
- **device** (``str``): The device where to perform computation

#### demo_trajectory.py

- **algo** (``str``) : Either "dcps", "ddrm", "dps", "mcgdiff", "pgdm", or "reddiff"
- **n_steps** (``int``) : The number of diffusion steps to use
- **std** (``float`` strictly positive) : the standard deviation of the noise
- **n_samples** (``int``): the number of samples to generate
- **idx_selected** (``List[int]``) : List of the indices of the trajectory to consider
- **device** (``str``): The device where to perform computation

The considered problems here are three trajectory completion problems with observed coordinates being: beginning, middle, and end.
Change the mask ``missing_coordinates`` to consider other completion problems.


The script ``training_trajectory.py`` can be used to train the diffusion model for trajectories on ``ucy`` dataset.


## Downloading checkpoints

- [Imagnet](https://github.com/openai/guided-diffusion)
- [FFHQ](https://github.com/DPS2022/diffusion-posterior-sampling)
- [Trajectories](https://drive.google.com/drive/folders/1gZb-kMX6TPuci7moDcwIMD15gfQLem7l?usp=share_link)
