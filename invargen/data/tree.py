from invargen.data.expression import *
from invargen.data.tokens import *


class ExpressionBuilder:
    stack: List[Expression]

    def __init__(self):
        self.stack = []

    def get_tree(self) -> Expression:
        if len(self.stack) == 1:
            return self.stack[0]
        else:
            raise InvalidExpressionException(f"Expected only one tree, got {len(self.stack)}")

    def add_token(self, token: Token):
        if not self.validate(token):
            raise InvalidExpressionException(f"Token {token} not allowed here, stack: {self.stack}.")
        if isinstance(token, OperatorToken):
            n_args: int = token.operator.n_args()
            children = []
            for _ in range(n_args):
                children.append(self.stack.pop())
            self.stack.append(token.operator(*reversed(children)))  # type: ignore
        elif isinstance(token, ConstantToken):
            self.stack.append(Constant(token.constant))
        elif isinstance(token, FeatureToken):
            self.stack.append(Feature(token.feature))
        else:
            assert False

    def is_valid(self) -> bool:
        return len(self.stack) == 1 and self.stack[0].is_featured

    def validate(self, token: Token) -> bool:
        if isinstance(token, OperatorToken):
            return self.validate_op(token.operator)
        elif isinstance(token, ConstantToken):
            return self.validate_const()
        elif isinstance(token, FeatureToken):
            return self.validate_feature()
        else:
            assert False

    def validate_op(self, op: Type[Operator]) -> bool:
        if len(self.stack) < op.n_args():
            return False
        if issubclass(op, UnaryOperator):
            if not self.stack[-1].is_featured:
                return False
        elif issubclass(op, BinaryOperator):
            if not self.stack[-1].is_featured and not self.stack[-2].is_featured:
                return False
        else:
            assert False
        return True

    def validate_const(self) -> bool:
        return len(self.stack) == 0 or self.stack[-1].is_featured

    def validate_feature(self) -> bool:
        return True


class InvalidExpressionException(ValueError):
    pass


if __name__ == '__main__':
    tokens = [
        FeatureToken(FeatureType.C),
        OperatorToken(Abs),
        FeatureToken(FeatureType.A),
        FeatureToken(FeatureType.B),
        OperatorToken(Add),
    ]

    builder = ExpressionBuilder()
    for token in tokens:
        builder.add_token(token)

    print(f'res: {str(builder.get_tree())}')
    print(f'ref: Add(Ref(Abs($low),-10),Div($high,$close))')
