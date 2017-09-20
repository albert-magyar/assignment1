import ast
import inspect
import astor
import textwrap
import operator as op
import random
from typing import List

########
## IR ##
########
"""
Expr = BinOp(Bop op, Expr left, Expr right)
     | CmpOp(Cop op, Expr left, Expr right)
     | UnOp(Uop op, Expr e)
     | Ref(Str name, Expr? index)
     | FloatConst(float val)
     | IntConst(int val)

Uop = Neg | Not
Bop = Add | Sub | Mul | Div | Mod | And | Or
Cop =  EQ |  NE |  LT |  GT |  LE | GE

Stmt = Assign(Ref ref, Expr val)
     | Block(Stmt* body)
     | If(Expr cond, Stmt body, Stmt? elseBody)
     | For(Str var, Expr min, Expr max, Stmt body)
     | Return(Expr val)
	 | FuncDef(Str name, Str* args, Stmt body)
"""

## Exprs ##
class BinOp(ast.AST):
    _fields = ['op', 'left', 'right']

class CmpOp(ast.AST):
    _fields = ['op', 'left', 'right']

class UnOp(ast.AST):
    _fields = ['op', 'e']

class Ref(ast.AST):
    _fields = ['name', 'index']

    def __init__(self, name, index=None):
        return super().__init__(name, index)

class IntConst(ast.AST):
    _fields = ['val',]

class FloatConst(ast.AST):
    _fields = ['val',]

## Stmts ##
class Assign(ast.AST):
    _fields = ['ref', 'val']

class Block(ast.AST):
    _fields = ['body',]

class If(ast.AST):
    _fields = ['cond', 'body', 'elseBody']

    def __init__(self, cond, body, elseBody=None):
        return super().__init__(cond, body, elseBody)

class For(ast.AST):
    _fields = ['var', 'min', 'max', 'body']

class Return(ast.AST):
    _fields = ['val',]

class FuncDef(ast.AST):
    _fields = ['name', 'args', 'body']

class PythonToSimple(ast.NodeVisitor):
    """
    Translate a Python AST to our simplified IR.

    TODO: Your first job is to complete this implementation.
    You only need to translate Python constructs which are actually
    representable in our simplified IR.

    As a bonus, try implementing logic to catch non-representable
    Python AST constructs and raise a `NotImplementedError` when you
    do. Without this, the compiler will just explode with an
    arbitrary error or generate malformed results. Carefully catching
    and reporting errors is one of the most challenging parts of
    building a user-friendly compiler.
    """
    def visit_Num(self, node):
        if isinstance(node.n, int):
            return IntConst(val=node.n)
        elif isinstance(node.n, float):
            return FloatConst(val=node.n)
        else:
            raise NotImplementedError("Unrecognized literal format")

    def visit_Str(self, node):
        raise NotImplementedError("Compiled code cannot contain string literals")

    def visit_Compare(self, node):
        op_dict = { ast.Eq: "EQ", ast.NotEq: "NE", ast.Lt: "LT", ast.LtE: "LE", ast.Gt: "GT", ast.GtE: "GE" }
        if (len(node.ops) != 1 or len(node.comparators) != 1):
            raise NotImplementedError("Compiled code cannot contain compound comparators")
        if (type(node.ops[0]) not in op_dict):
            raise NotImplementedError("Unrecognized comparison operator")
        return CmpOp(op_dict[type(node.ops[0])], self.visit(node.left), self.visit(node.comparators[0]))

    def visit_BinOp(self, node):
        op_dict = { ast.Add: "Add", ast.Sub: "Sub", ast.Mult: "Mul", ast.Div: "Div", ast.Mod: "Mod" }
        if (type(node.op) not in op_dict):
            raise NotImplementedError("Unrecognized binary operator")
        return BinOp(op_dict[type(node.op)], self.visit(node.left), self.visit(node.right))

    def visit_BoolOp(self, node):
        op_dict = { ast.And: "And", ast.Or: "Or" }
        if (len(node.values) != 2):
            raise NotImplementedError("Unrecognized boolean expression")
        if (type(node.op) not in op_dict):
            raise NotImplementedError("Unrecognized boolean operator")
        return BinOp(op_dict[type(node.op)], self.visit(node.values[0]), self.visit(node.values[1]))

    def visit_UnaryOp(self, node):
        op_dict = { ast.Not: "Not", ast.USub: "Neg" }
        if (type(node.op) not in op_dict):
            raise NotImplementedError("Unrecognized unary operator")
        return UnOp(op_dict[type(node.op)], self.visit(node.operand))

    def visit_Name(self, node):
        return Ref(node.id, None)

    def visit_Subscript(self, node):
        assert isinstance(node.value, ast.Name)
        assert isinstance(node.slice, ast.Index)
        return Ref(node.value.id, self.visit(node.slice.value))

    def visit_Assign(self, node):
        if (len(node.targets) != 1):
            raise NotImplementedError("Compiled code cannot contain multiple-variable assignment statements")
        return Assign(self.visit(node.targets[0]), self.visit(node.value))

    def visit_If(self, node):
        ifBlock = Block([self.visit(s) for s in node.body])
        elseBlock = Block([self.visit(s) for s in node.orelse]) if (len(node.orelse) > 0) else None
        return If(self.visit(node.test), ifBlock, elseBlock)

    def process_range(self, node):
        assert isinstance(node, ast.Call)
        assert isinstance(node.func, ast.Name)
        assert (node.func.id == "range")
        if (len(node.args) == 1):
            return (IntConst(0), self.visit(node.args[0]))
        elif (len(node.args) == 2):
            return (self.visit(node.args[0]), self.visit(node.args[1]))
        else:
            raise NotImplementedError("Compiled for loops must use 2-argument integral ranges")

    def visit_For(self, node):
        if (len(node.orelse) > 0):
            raise NotImplementedError("Compiled code cannot contain for loops with break statements")
        if (not isinstance(node.target, ast.Name)):
            raise NotImplementedError("Compiled for loops must have a scalar iteration variable")
        varMin, varMax = self.process_range(node.iter)
        body = Block([self.visit(s) for s in node.body])
        return For(self.visit(node.target).name, varMin, varMax, body)

    def visit_Return(self, node):
        return Return(self.visit(node.value))

    def visit_FunctionDef(self, func):
        assert isinstance(func.body, list)
        args = [a.arg for a in func.args.args]
        body = Block([self.visit(stmt) for stmt in func.body])
        return FuncDef(func.name, args, body)

def Interpret(ir, *args):
    assert isinstance(ir, FuncDef)
    assert len(args) == len(ir.args)
    syms = {}
    for (a, v) in zip(ir.args, args):
        syms[a] = v

    class EvalExpr(ast.NodeVisitor):
        def __init__(self, symbolTable):
            self.syms = symbolTable

        def visit_IntConst(self, node):
            return node.val

        def visit_FloatConst(self, node):
            return node.val

        def visit_Ref(self, node):
            if node.index is not None:
                return self.syms[node.name][self.visit(node.index)]
            else:
                return self.syms[node.name]

        def visit_BinOp(self, node):
            fn_dict = { "Add": op.add, "Sub": op.sub, "Mul": op.mul, "Div": op.truediv, "Mod": op.mod, "And": op.__and__, "Or": op.__or__ }
            return fn_dict[node.op](self.visit(node.left), self.visit(node.right))

        def visit_CmpOp(self, node):
            fn_dict = { "EQ": op.eq, "NE": op.ne, "LT": op.lt, "LE": op.le, "GT": op.gt, "GE": op.ge }
            return fn_dict[node.op](self.visit(node.left), self.visit(node.right))

        def visit_UnOp(self, node):
            fn_dict = { "Not": op.__not__, "Neg": op.neg }
            return fn_dict[node.op](self.visit(node.e))

    class ReturnValue:
        def __init__(self, val):
            self.val = val

    class EvalStmt(ast.NodeVisitor):
        def __init__(self, symbolTable):
            self.syms = symbolTable
            self.evaluator = EvalExpr(syms)

        def visit_Assign(self, node):
            rhs = self.evaluator.visit(node.val)
            if node.ref.index is not None:
                self.syms[node.ref.name][self.evaluator.visit(node.ref.index)] = rhs
            else:
                self.syms[node.ref.name] = rhs

        def visit_Block(self, node):
            for s in node.body:
                result = self.visit(s)
                if (isinstance(result, ReturnValue)):
                    return result

        def visit_For(self, node):
            start = self.evaluator.visit(node.min)
            end = self.evaluator.visit(node.max)
            for i in range(start, end):
                self.syms[node.var] = i
                result = self.visit(node.body)
                if (isinstance(result, ReturnValue)):
                    return result

        def visit_If(self, node):
            if (self.evaluator.visit(node.cond)):
                return self.visit(node.body)
            elif node.elseBody is not None:
                return self.visit(node.elseBody)

        def visit_Return(self, node):
            return ReturnValue(self.evaluator.visit(node.val))

    evaluator = EvalStmt(syms)
    result = evaluator.visit(ir.body)
    assert isinstance(result, ReturnValue)
    return result.val

def Compile(f):
    """'Compile' the function f"""
    # Parse and extract the function definition AST
    fun = ast.parse(textwrap.dedent(inspect.getsource(f))).body[0]
    print("Python AST:\n{}\n".format(astor.dump(fun)))

    simpleFun = PythonToSimple().visit(fun)

    print("Simple IR:\n{}\n".format(astor.dump(simpleFun)))

    # package up our generated simple IR in a
    def run(*args):
        return Interpret(simpleFun, *args)

    return run


#############
## TEST IT ##
#############

# Define a fibonacci_list test program to start
def fibonacci_list(n, nums) -> List[int]:
    for i in range(0, n):
        if (nums[i] > 0):
            a = 0
            b = 1
            for j in range(0, nums[i]):
                b = a + b
                a = b - a
            nums[i] = a
        else:
            nums[i] = 0
    maxval = 0
    for i in range(0, n):
        if (nums[i] > maxval):
            maxval = nums[i]
    return maxval

def quadratic_matrix_vec(a, x, tmp, m, n) -> int:
    # A is (m, n) -- row-major
    # x is (n, 1)
    # xT * AT * A * x
    # do matmul first for fun
    # do everything in a silly way to test more stuff
    for i in range(0, n):
        for j in range(0, n):
            tmp[n * i + j] = 0.0
    for i in range(0, n):
        for j in range(0, n):
            for k in range(0, m):
                aT_idx_in_a = k * n + i
                a_idx = k * n + j
                tmp_idx = (aT_idx_in_a % n) * n + (a_idx % n)
                tmp[tmp_idx] = tmp[tmp_idx] + ((a[aT_idx_in_a] * a[a_idx] * 2.0) / 2.0)
    for i in range(0, n):
        sumval = 0
        for j in range(0, n):
            sumval = sumval + (tmp[i * n + j] * x[j])
        tmp[i] = sumval
    for i in range(0, n):
        tmp[i] = tmp[i] * x[i]
    retval = 0
    for i in range(0, n):
        retval = retval + tmp[i]
    return retval

def every_op(a, b, c, d):
    if (a == b):
        return (a + b + c + d) / ((a - b) + (-c))
    if (a <= b and c > d):
        return a / (d / c)
    if (a >= b or c < d):
        return a + b * c
    if (a < b and (c < a)):
        return (a % b) * d
    if (a <= b and not (c > d)):
        return 1
    return (a / (b / (c / d))) + 45.0

def test_it():
    arg_array_test = [0, 4, 8, 7, 3, 9, 11, 12, 3, 19, -4]
    arg_array_check = [0, 4, 8, 7, 3, 9, 11, 12, 3, 19, -4]
    fibonacci_list_interpreted = Compile(fibonacci_list)
    maxFibTest = fibonacci_list_interpreted(len(arg_array_test), arg_array_test)
    maxFibCheck = fibonacci_list(len(arg_array_check), arg_array_check)
    assert maxFibTest == maxFibCheck
    assert arg_array_test == arg_array_check

    a_test = [1.0, 2.3, 4.9, 0.7, 13.2, 6.0]
    x_test = [4.4, 0.8, 1.0]
    tmp_test = [0.0] * 9
    quadratic_matrix_vec_interpreted = Compile(quadratic_matrix_vec)
    result_test = quadratic_matrix_vec_interpreted(a_test, x_test, tmp_test, 2, 3)

    tmp_check = [0.0] * 9
    result_check = quadratic_matrix_vec(a_test, x_test, tmp_check, 2, 3)
    assert result_test == result_check

    every_op_interpreted = Compile(every_op)
    for a in range(0, 10):
        for b in range(0, 10):
            for i in range(0, 100):
                c = random.uniform(-100.0, 100.0)
                d = random.uniform(-100.0, 100.0)
                result_test = every_op_interpreted(a, b, c, d)
                result_check = every_op(a, b, c, d)
                assert result_test == result_check

if __name__ == '__main__':
    test_it()
