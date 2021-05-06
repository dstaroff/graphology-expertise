from utils.globals import (
    np,
    pywt,
)
from utils.plot import plot_image


def get_wavelet_transform_coefficients(image, verbose=False) -> (float, float, float, float):
    '''
    Calculates wavelet coefficients of an image via wavelet transform

    Returns:
    Tuple (
        Approximate coefficient [low freq],
        Horizontal detailed coefficient,
        Vertical detailed coefficient,
        Diagonal detailed coefficient [high freq]
    )
    '''
    image = np.float32(image) / 255
    cA, (cH, cV, cD) = pywt.dwt2(image, wavelet='db4', mode='periodization')

    if verbose:
        plot_image(cA, title='cA')

    return cA, (cH, cV, cD)
