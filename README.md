# Imperfect-Information Alpha Zero
This is a placeholder README. We'll update this later, once the report is written.

### Docker Installation
For easy environment setup, we can use [nvidia-docker](https://github.com/NVIDIA/nvidia-docker). Once you have nvidia-docker set up, we can then simply run:
```
./setup_env.sh
```
to set up a (default: pyTorch) Jupyter docker container. We can now open a new terminal and enter:
```
docker exec -ti pytorch_notebook python main.py
```

### Experiments
Obviously haven't been finished yet.

The original paper can be found [here](https://github.com/suragnair/alpha-zero-general/raw/master/pretrained_models/writeup.pdf). We'll soon replace it with our own.

### Contributing
Will depend on what actually gets done! Some ideas:
* More imperfect-info games - plenty to modify on original repo
* Making games work with human players
* Making MCTS work with non-placement games (Kriegspiel & such) - important if adding new games
* Games that allow one agent imperfect, one perfect (obviously one-sided but could lead to interesting results). May demand a new framework with only one class per game

### Contributors and Credits
* [Surag Nair](https://github.com/suragnair) and the other contributors to [the original alpha-zero-general repository](https://github.com/suragnair/alpha-zero-general/). Their code has been reused under the MIT licence.
* [Kobe Knowles](https://github.com/Kobster2434) adjusted the neural networks, trained the models and obtained the results seen in the paper.
* Other team members' work in other repos, will link once they're posted