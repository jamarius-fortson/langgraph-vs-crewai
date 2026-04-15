import ast
import operator as op

# Simple safe evaluator for math expressions
operators = {ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul,
             ast.Div: op.truediv, ast.Pow: op.pow, ast.BitXor: op.xor,
             ast.USub: op.neg}

def eval_expr(expr: str):
    return eval_(ast.parse(expr, mode='eval').body)

def eval_(node):
    if isinstance(node, ast.Num): # <python 3.8
        return node.n
    elif isinstance(node, ast.Constant): # >= python 3.8
        return node.value
    elif isinstance(node, ast.BinOp):
        return operators[type(node.op)](eval_(node.left), eval_(node.right))
    elif isinstance(node, ast.UnaryOp):
        return operators[type(node.op)](eval_(node.operand))
    else:
        raise TypeError(node)

def calculator(expression: str) -> float:
    """Mock calculator tool for benchmarking."""
    try:
        result = eval_expr(expression)
        return {"result": float(result)}
    except Exception as e:
        return {"error": str(e)}
