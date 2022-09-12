# Combining Variational Autoencoders and Transformer Language Models for Improved Password Generation

Code to train a variational autoencoder with GPT2-based encoder and decoder on password data, as in

D. Biesner, K. Cvejoski, and R. Sifa, ‘Combining Variational Autoencoders and Transformer Language Models for Improved Password Generation’, in ARES 2022: The 17th International Conference on Availability, Reliability and Security.

[Researchgate](https://www.researchgate.net/publication/362873332_Combining_Variational_Autoencoders_and_Transformer_Language_Models_for_Improved_Password_Generation)

[acm.org pdf](https://dl.acm.org/doi/pdf/10.1145/3538969.3539000)

Bibtex:
```
@inproceedings{DBLP:conf/IEEEares/BiesnerCS22,
  author    = {David Biesner and
               Kostadin Cvejoski and
               Rafet Sifa},
  title     = {Combining Variational Autoencoders and Transformer Language Models
               for Improved Password Generation},
  booktitle = {{ARES} 2022: The 17th International Conference on Availability, Reliability
               and Security, Vienna,Austria, August 23 - 26, 2022},
  pages     = {37:1--37:6},
  publisher = {{ACM}},
  year      = {2022},
  url       = {https://doi.org/10.1145/3538969.3539000},
  doi       = {10.1145/3538969.3539000},
  timestamp = {Fri, 19 Aug 2022 10:16:29 +0200},
  biburl    = {https://dblp.org/rec/conf/IEEEares/BiesnerCS22.bib},
}
```

Contact:
David Biesner
<david.biesner@iais.fraunhofer.de>


## Installation

Install the requirements and the package by
```
pip install -r requirements.txt
pip install -e .
```

Or create a new conda environment
```
conda env create -f environment.yml
```


## Download password data

Use download script to download some password datasets or use your own.
Experiments in the paper require the `rockyou` dataset (~140MB):
```
python scripts/download_raw_data.py --datasets rockyou --output data/
```


## Train a model

Use a config.yaml file and the training script to train a new model:
```
python scripts/train.py --config configs/vae_gpt2.yaml
```

You will need to adjust the config.yaml file for your system.

Update input and output paths:
```
data_path: &DATA_PATH ~/password_generation/data/rockyou.txt
logging_dir: &LOGGING_DIR ~/password_generation/logging/
checkpoints_dir: &CHECKPOINTS_DIR ~/password_generation/checkpoints/
```

Disable [wandb-logging](https://www.wandb.ai) to only use tensorboard:
```
logging:
  use_wandb: True
  logging_dir: *LOGGING_DIR
```

Use `in_memory` to load the entire dataset into memory before training and `use_cache` to cache the tokenization process:
```
in_memory: &IN_MEMORY True
use_cache: &USE_CACHE True
```
Cached password tokens are stored in the same directory as the password.txt file.


## Generate data

To generate data from a model checkpoint, use the generation script:
```
python scripts/generate.py -m checkpoints/vae_gpt2/run_timestamp/model.pth --config configs/vae_gpt2.yaml --num-passwords 100000 --batch-size 1000
```
Model checkpoint must match the model definition in the config.yaml file!
