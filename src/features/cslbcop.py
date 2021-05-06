from utils.globals import (
    feature,
    np,
)
from utils.plot import plot_image

ANGLES = (
    0,
    45,
    90,
    135
)


def get_CSLBCoP_comatrix(image, verbose=False) -> np.array:
    '''
    Calculates co-matrix Center Symmetric Local Binary Co-occurrence Pattern

    Returns:
        CSLBCoP comatrix
    '''
    lbp = np.uint8(
        feature.local_binary_pattern(
            image=image,
            P=4,
            R=1,
        )
    )
    if verbose:
        plot_image(lbp, 'Local Binary Pattern of line')

    res = feature.greycomatrix(
        image=lbp,
        distances=[1],
        angles=ANGLES,
        levels=16,
    )

    return res
