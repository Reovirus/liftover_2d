from src.pixel_dividers.base_divider import BaseDivider

class LinearModelDivider(BaseDivider):
    """
    Divider that divides the image into pixels using a linear model.
    This class should be inherited by all pixel divider implementations that use a linear model.
    """
    def __init__(
        self,
        mode='resample', 
        cvd_norm=None, 
        random_seed=None
    ):
        self.mode = mode,
        self.cvd_norm = cvd_norm
        self.random_seed = random_seed

    def _divide_pixel(self, old_nums, old_borders, new_borders):
        # Implement the logic for dividing a pixel using a linear model
        return 

    def divide_from_table(self, table_to_divide, pixels_table):
        # Implement the logic for dividing the image into pixels from the table
        pass