from utils.globals import (
    cv2,
    plt,
)

DPI = 72


def plot_image(image, title=None):
    if len(image.shape) > 2:
        height, width, _ = image.shape
    else:
        height, width = image.shape
    height = height / float(DPI)
    width = width / float(DPI)

    plt.figure(dpi=DPI, figsize=(width, height))

    if title is not None:
        plt.title(title)

    # noinspection PyBroadException
    try:
        plt.imshow(cv2.cvtColor(image.copy(), cv2.COLOR_BGR2RGB))
    except:
        plt.imshow(image.copy() / 255, cmap='nipy_spectral')

    plt.show()
