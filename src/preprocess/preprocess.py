from features import (
    get_wavelet_transform_coefficients,
    get_CSLBCoP_comatrix,
)
from utils import (
    logger,
)
from utils.globals import (
    cv2,
    np,
    PCA,
    MinMaxScaler,
)
from utils.plot import (
    plot_image,
)


def grayscale(image, verbose=False):
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if verbose:
        plot_image(grayscale_image, 'Grayscale image')

    return grayscale_image


def threshold(image, verbose=False):
    _, thresholded_image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    if verbose:
        plot_image(thresholded_image, 'Thresholded image')

    return thresholded_image


def preprocess(image, verbose=False):
    return threshold(grayscale(image, verbose=verbose), verbose=verbose)


def discard(image):
    image = np.uint8(image)
    _, image_label, statistics, _ = cv2.connectedComponentsWithStats(image, connectivity=4)
    msk = np.isin(image_label, np.where(statistics[:, cv2.CC_STAT_WIDTH] > 500)[0])
    image[msk] = 0

    return image


def get_lines_coordinates(image, verbose=False):
    image_without_lines = discard(image.copy())
    if verbose:
        plot_image(image_without_lines, 'Image without lines')

    image_with_just_lines = cv2.bitwise_xor(image, image_without_lines)
    if verbose:
        plot_image(image_with_just_lines, 'Image with just lines')

    sum_horizontal = np.sum(image_with_just_lines, axis=1)
    lines_indexes = np.argwhere(sum_horizontal > 2000).flatten()
    y_start = lines_indexes[np.argmax(lines_indexes > (lines_indexes[0] + 150))]
    y_end = max(lines_indexes)

    # closing
    kernel_closing = np.ones((5, 200), np.uint8)
    img_closing = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel_closing)
    if verbose:
        plot_image(img_closing, 'Closing image')

    # opening
    kernel_opening = np.ones((1, 200), np.uint8)
    img_opening = cv2.morphologyEx(img_closing, cv2.MORPH_OPEN, kernel_opening)
    if verbose:
        plot_image(img_opening, 'Opening image')

    contours, _ = cv2.findContours(img_opening.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # sort contours
    sorted_contours = sorted(contours, key=lambda contour: cv2.boundingRect(contour)[0])
    line_coordinates = []
    for contour in sorted_contours:
        x, y, w, h = cv2.boundingRect(contour)
        if y_end > y > y_start and h > 25:
            line_coordinates.append((y, y + h, x, x + w))

    return line_coordinates


def get_lines(image, verbose=False):
    preprocessed_image = preprocess(image, verbose=verbose)

    lines_coordinates = get_lines_coordinates(preprocessed_image, verbose=verbose)
    lines = []
    for (y1, y2, x1, x2) in lines_coordinates:
        lines.append(preprocessed_image[y1:y2, x1:x2])

    return lines


def get_data_for_image(image, verbose=False):
    X_list = []

    lines_list = get_lines(image, verbose=verbose)
    for line_number, line in enumerate(lines_list):
        cA, (_, _, _) = get_wavelet_transform_coefficients(line, verbose=verbose)

        histogram_of_line = get_CSLBCoP_comatrix(cA, verbose=verbose).flatten()

        X_list.append(histogram_of_line)

    return np.array(X_list)


def pca_transform(X, min_components=32, verbose=False):
    scaler = MinMaxScaler(feature_range=(-1, 1))
    X = scaler.fit_transform(X)

    components = min(min_components, X.shape[1])
    if verbose:
        logger.info(f'Number of components={components}')

    pca = PCA(n_components=components, copy=False)
    X = pca.fit_transform(X)
    X = scaler.fit_transform(X)

    if verbose:
        logger.info(f'New data shape: {X.shape}')

    return X
