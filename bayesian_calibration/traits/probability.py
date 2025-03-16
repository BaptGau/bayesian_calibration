from pydantic.v1 import ConstrainedFloat


class Probability(ConstrainedFloat):
    ge = 0.0
    le = 1.0
