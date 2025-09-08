from scipy.ndimage import zoom
import numpy as np

from src.pixel_dividers.base_divider import BaseNeigborUsingDivider


class PolynimialSplineDivider(BaseNeigborUsingDivider):
    METRIC_NAME='POLYNOM_2D_SPLINE'
    NEED_NORMALISATION=False
    NEED_SAMPLING=True
    COMPUTE_COUNTS=False

    def __init__(self, mode, ident: int=3, scale_factor: int=20, k=4):
        self.__k = k
        super().__init__(ident=ident, scale_factor=scale_factor, mode=mode)


    def _process_one_window(self, counts_arr, location):
        zoomed = zoom(counts_arr, zoom=self._scale_factor, order=self.__k, grid_mode=True, mode='nearest')
        our_pixel = zoomed[
            location[0]*self._scale_factor:(location[0]+1)*self._scale_factor, 
            location[1]*self._scale_factor:(location[1]+1)*self._scale_factor, 
        ]
        our_pixel = np.clip(our_pixel, 0, None)
        s = our_pixel.sum()
        if s > 0:
            our_pixel /= s
        return our_pixel