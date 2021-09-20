# picoGAN 🧠
A very small GAN for unsupervised learning

Two simple models to demonstrate both a traditional dense neural network, and a generative adversarial network (GAN) for fraud detection.
The dataset contains preprocessed banknote images, reduced to four numeric features and a single label. Refer to the [original data source](https://archive.ics.uci.edu/ml/datasets/banknote+authentication#) for further details.

## The Models

### naive.py
A very small fully connected neural network for learning classification on the unaltered balanced dataset. The network of (4 > 16 > 1) nodes trains to ~99% accuracy in 10 epochs.

Fairly straightforward.

### gan.py
A very small GAN for learning classification on the legitimate half of the dataset only! By constructing two neural networks, we set up a training competition between them. The generator learns to create banknotes from noise, while the discriminator learns to tell real banknotes apart from the generator's fake output.

After 600 epochs, the generator's banknotes look pretty real to the discriminator, but the discriminator is getting pretty good at telling the real from the fake!

Can a neural network learn to identify fraudulent banknotes without ever seeing one? Run gan.py to find out!

## Usage
1. Install Python 3 if not already
2. Create/activate a virtual environment and install deps:
```bash
$ cd picoGan/
$ python3 -m venv venv
$ source venv/bin/activate
(venv) $ pip install -r requirements.txt
```
3. Run the demos
```bash
(venv) $ python naive.py
(venv) $ python gan.py
```
4. Tweak and experiment!
