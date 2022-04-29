import hydra
from hydra.utils import get_original_cwd
from omegaconf import DictConfig
import matplotlib.pyplot as plt
import cifar10
import fmnist
import mnist


def plot_pixel_val_histogram(data, name):
    plt.hist(data.flatten(), bins=256)
    plt.yscale('log')
    plt.xlabel("Pixel Value")
    plt.ylabel("Count")
    plt.ylim((0, 10000000))
    plt.title(name)
    plt.show()


@hydra.main(config_path="conf", config_name="config_binary")
def main(cfg: DictConfig):
    x_mnist = mnist.load(get_original_cwd())[0]
    x_fmnist = fmnist.load(get_original_cwd())[0]
    x_cifar10_color = cifar10.load(get_original_cwd(), method='color')[0]
    x_cifar10_average = cifar10.load(get_original_cwd(), method='average')[0]
    x_cifar10_grayscale = cifar10.load(get_original_cwd(), method='grayscale')[0]

    plot_pixel_val_histogram(x_mnist, "Pixel Value Distribution for MNIST")
    plot_pixel_val_histogram(x_fmnist, "Pixel Value Distribution for FASHION")
    plot_pixel_val_histogram(x_cifar10_color, "Pixel Value Distribution for CIFAR10 (COLOR)")
    plot_pixel_val_histogram(x_cifar10_average, "Pixel Value Distribution for CIFAR10 (AVERAGED)")
    plot_pixel_val_histogram(x_cifar10_grayscale, "Pixel Value Distribution for CIFAR10 (GRAYSCALE)")


if __name__ == "__main__":
    main()
