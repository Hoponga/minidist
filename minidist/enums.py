from enum import Enum


class ReduceOp(Enum):
    SUM = "sum"
    PRODUCT = "product"
    MIN = "min"
    MAX = "max"
    AVG = "avg"
