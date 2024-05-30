'''MNIST をダウンロードし表示する。

Reference
https://github.com/enakai00/colab_jaxbook/blob/main/Chapter02/
2.%20MNIST%20dataset%20sample.ipynb
'''
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.datasets import mnist


def main():

    (
       (train_images, train_labels),
       (test_images, test_labels)
    ) = mnist.load_data()

    train_images = train_images.reshape([-1, 784]).astype('float32') / 255
    test_images = test_images.reshape([-1, 784]).astype('float32') / 255

    train_labels = np.eye(10)[train_labels]
    test_labels = np.eye(10)[test_labels]

    fig = plt.figure(figsize=(8, 4))
    for c, (image, label) in enumerate(zip(train_images[:10], train_labels)):
        subplot = fig.add_subplot(2, 5, c+1)
        subplot.set_xticks([])
        subplot.set_yticks([])
        subplot.set_title(np.argmax(label))
        subplot.imshow(image.reshape([28, 28]),
                       vmin=0, vmax=1,
                       cmap=plt.cm.gray_r)

    plt.show()
    plt.close()


if __name__ == '__main__':
    main()
