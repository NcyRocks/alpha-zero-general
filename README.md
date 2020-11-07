# Imperfect-Information Alpha Zero
This repository is an adaptation of alpha-zero-general containing information-imperfect board games variants, and other code adjusted to allow for information imperfection.

The corresponding paper can be found [here](https://github.com/NcyRocks/impinf-alphazero/blob/master/pretrained_models/Adapting%20AlphaZero%20to%20Imperfect%20Information.pdf).

### Docker Installation
For easy environment setup, we can use [nvidia-docker](https://github.com/NVIDIA/nvidia-docker). Once you have nvidia-docker set up, we can then simply run:
```
./setup_env.sh
```
to set up a (default: pyTorch) Jupyter docker container. We can now open a new terminal and enter:
```
docker exec -ti pytorch_notebook python main.py
```

### Contributing
Some contributions that would be interesting to see:
* More imperfect-info games
* Making games work with human players
* Making MCTS work with non-placement games (Kriegspiel & such) - important if adding new games
* Games that allow one agent imperfect, one perfect (obviously one-sided but could lead to interesting results)

### Contributors and Credits
* [Surag Nair](https://github.com/suragnair) and the other contributors to [the original alpha-zero-general repository](https://github.com/suragnair/alpha-zero-general/). Their code has been reused under the MIT licence.
* [Kobe Knowles](https://github.com/Kobster2434) adjusted the neural networks, trained the models and obtained most of the results seen in the paper.
* [Bhargava Gowda](https://github.com/BhargavaGowda) adapted MuZero in a similar way. His repository can be found [here](https://github.com/BhargavaGowda/muzero-general).
