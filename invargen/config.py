from invargen.data.expression import *


MAX_EXPR_LENGTH = 20
MAX_EPISODE_LENGTH = 256

OPERATORS = [
    # Unary
    Abs, Sign, Log,
    # Binary
    Add, Sub, Mul, Div, Pow, Greater, Less
]

CONSTANTS = [int(i) for i in range(1, 1000)]

REWARD_PER_STEP = 0.
