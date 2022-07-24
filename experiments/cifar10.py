import matplotlib.pyplot as plt
from torch.utils import data

from datasets.cifar10_dataloader import Cifar10Dataloader
from layers import VisionTransformer

TRAIN_DATA_PATH = "/home/filipe/Downloads/research/cifar-10/train/"
BATCH_SIZE = 256


def main():
    cifar_data = Cifar10Dataloader("/home/filipe/Downloads/research/cifar-10/train/",
                                   "/home/filipe/Downloads/research/cifar-10/trainLabels.csv", "label", (3, 3))

    test_data_loader = data.DataLoader(
        cifar_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=1)

    net = VisionTransformer(10, 300, 10)
    net.cuda()

    for _data in test_data_loader:
        _data = _data.float()
        _data = _data.cuda()

        res = net.forward(_data)


def plot_cifar_example(test_data_loader):
    for _data in test_data_loader:
        plot_images(_data[:, 1:], (3, 3), figsize=(2, 2), img_shape_restore=(3, 10, 10))
        plot_images(_data[:, 1:], (1, 9), figsize=(10, 10), img_shape_restore=(3, 10, 10))
        break


def plot_images(tensors, shape, figsize, img_shape_restore):
    # fig, axs = plt.subplots(*shape)

    fig = plt.figure(figsize=figsize)  # Notice the equal aspect ratio
    ax = [fig.add_subplot(*shape, i + 1) for i in range(shape[0] * shape[1])]

    tensors = tensors[0]

    for i in range(shape[0]):
        for j in range(shape[1]):
            idx = i * shape[1] + j
            img = tensors[idx].reshape(img_shape_restore)

            ax[idx].get_xaxis().set_visible(False)
            ax[idx].get_yaxis().set_visible(False)
            ax[idx].imshow(img.permute(1, 2, 0))

    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    plt.show()


if __name__ == '__main__':
    main()
