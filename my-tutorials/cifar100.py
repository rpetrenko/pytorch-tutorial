"""
CIFAR100 dataset playground, url: https://www.cs.toronto.edu/~kriz/cifar.html

Couldn't now figure out labels mapping. some images showed wrong labels,
like keyboard was labeled as kangaroo, and cups and crocodile had to be swapped
"""
import time
import torch
import torchvision
import torch.nn as nn
import numpy as np
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as dsets
from torch.autograd import Variable
import matplotlib.pyplot as plt

from common.pytorch_helpers import DATA_PATH


LABELS = [
    ("aquatic mammals", "beaver, dolphin, otter, seal, whale"),
    ("fish", "aquarium fish, flatfish, ray, shark, trout"),
    ("flowers", "orchids, poppies, roses, sunflowers, tulips"),
    ("food containers", "bottles, bowls, cans, cups, plates"),
    ("fruit and vegetables", "apples, mushrooms, oranges, pears, sweet peppers"),
    ("household electrical devices", "clock, computer keyboard, lamp, telephone, television"),
    ("household furniture", "bed, chair, couch, table, wardrobe"),
    ("insects", "bee, beetle, butterfly, caterpillar, cockroach"),
    ("large carnivores", "bear, leopard, lion, tiger, wolf"),
    ("large man-made outdoor things", "bridge, castle, house, road, skyscraper"),
    ("large natural outdoor scenes", "cloud, forest, mountain, plain, sea"),
    ("large omnivores and herbivores", "camel, cattle, chimpanzee, elephant, kangaroo"),
    ("medium-sized mammals", "fox, porcupine, possum, raccoon, skunk"),
    ("non-insect invertebrates", "crab, lobster, snail, spider, worm"),
    ("people", "baby, boy, girl, man, woman"),
    ("reptiles", "crocodile, dinosaur, lizard, snake, turtle"),
    ("small mammals", "hamster, mouse, rabbit, shrew, squirrel"),
    ("trees", "maple, oak, palm, pine, willow"),
    ("vehicles 1", "bicycle, bus, motorcycle, pickup truck, train"),
    ("vehicles 2", "lawn-mower, rocket, streetcar, tank, tractor")
]


def create_fine_labels():
    res = []
    for l in LABELS:
        for a in l[1].split(','):
            res.append(a.strip())
    res = sorted(res)
    # swap cups and crocodile
    res[28], res[29] = res[29], res[28]
    return res


def print_fine_labels(fine_labels):
    i = 0
    for f in fine_labels:
        print i, f
        i += 1
    print


fine_labels = create_fine_labels()
print_fine_labels(fine_labels)

train_dataset = dsets.CIFAR100(root=DATA_PATH,
                               train=True,
                               transform=transforms.ToTensor(),
                               download=False)
plt.show()
for i in range(len(train_dataset)):
    # i = np.random.choice(range(len(train_dataset)))
    image, label = train_dataset[i]
    if True:
        plt.imshow(image.numpy().transpose(1,2,0))
        print label, fine_labels[label]
        pass
