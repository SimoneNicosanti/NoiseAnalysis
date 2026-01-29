from enum import Enum


class QuantizationVariants(Enum):
    NOT_QUANT = 0
    INT8_QUANT = 1
    INT4_QUANT = 2