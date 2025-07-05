from .base_divider import BaseDivider


class CVDNorm(BaseDivider):
    def __init__(self, cis, trans, mode):
        self.__cis = cis
        self.__trans = trans

    def _compute_weights(self, bin_remapper, bin_contacts):
        pass