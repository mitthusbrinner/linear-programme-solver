import math
from typing import List
from enum import Enum


class objective(Enum):
    """
    The objective in the objective function; either maximize or minimize.
    """
    maximize = 1
    minimize = 0


class inequality(Enum):
    """
    The relationship in an inequality, either geq (>=), equal or leq (<=).
    """
    leq = -1
    equal = 0
    geq = 1


class constraint:
    """
    Represents a constraint by a coefficient array, an inequality and a constant.

    Example: 4x_1 + 6x_3 <= 3 gets represented by [4, 0, 6], inequality.leq and 3.
    """

    def __init__(self, coefficients: List[float], relation: inequality, value: float) -> None:
        self.coefficients = coefficients
        self.relation = relation
        self.value = value

    def __repr__(self) -> str:
        output = []
        for i, c in enumerate(self.coefficients):
            if c == 0:
                continue
            output.append('{}x_{}'.format(c, i))
        output = " + ".join(output)
        output = output.replace("1x", "x")
        output = output.replace("+ -", "- ")
        if self.relation is inequality.leq:
            output += " <= "
        elif self.relation is inequality.geq:
            output += " >= "
        else:
            output += " = "
        output += (str(self.value))
        return output


class objectiveFunction:
    """
    Represents an objective function by a coefficient array and an objective.

    Example: Maximize 3x_1 + 4x_2 - 34x_3 gets represented by [3, 4, -32]
    and objective.maximize.
    """

    def __init__(self, coefficients: List[float], goal: objective) -> None:
        if not coefficients:
            raise Exception("Objective function cannot have zero variables")
        if not objective:
            raise Exception("Objective function must have an objective")
        self.coefficients = coefficients
        self.objective = goal

    def __repr__(self) -> str:
        goal = "maximize: \n\tZ = " if self.objective is objective.maximize else "minimize: \n\tZ = "
        variables = []
        for i, c in enumerate(self.coefficients):
            if c == 0:
                continue
            variables.append('{}x_{}'.format(c, i))
        variables = " + ".join(variables)
        variables = variables.replace("1.0x", "x")
        variables = variables.replace("+ -", "- ")
        return goal + "".join(variables)


class linearProgram:
    """
    Represents a linear program by a objective function and a list of constraints.
    """

    def __init__(self, objective_function: objectiveFunction, constraints: [constraint],
                 free_variables: List[int] = None) -> None:
        if free_variables is None:
            free_variables = []
        self.objective_function = objective_function
        self.constraints = constraints
        self.free_variables = free_variables
        self._is_valid_linear_program()

    def _is_valid_linear_program(self) -> None:
        if len(self.constraints) == 0:
            raise Exception("Linear program must have at least one constraint")
        if max(self.free_variables, default=-1) > len(self.objective_function.coefficients) - 1:
            raise Exception("Variables that do not exist are set to free")
        for constr in self.constraints:
            if len(constr.coefficients) != len(self.objective_function.coefficients):
                raise Exception("Constraints and the objective function must have an equal amount of variables")

    def __repr__(self) -> str:
        output_String = [str(self.objective_function)]
        constraints = []
        for constr in self.constraints:
            constraints.append("\t" + str(constr))
        constraints = "\n".join(constraints)
        constraints = constraints.replace("1.0x", "x")
        constraints = constraints.replace("+ -", "- ")
        output_String.append("\nsubject to: \n" + constraints + "\n")
        if not self.free_variables:
            non_free_variables = []
            for i in [k for k in range(len(self.objective_function.coefficients)) if k not in set(self.free_variables)]:
                non_free_variables.append("x_{}".format(i))
            non_free_variables = ", ".join(non_free_variables)
            output_String.append("\t" + non_free_variables + " >= 0\n")
        return "".join(output_String)


def solve(lp: linearProgram, display=False) -> List[float]:
    """
    Solves a linear program using the simplex algorithm.

    :param lp: a linear program.
    :param display: print the solution in a human-readable way if set to True else not.
    :return: solution to the linear program on the form  [Z x1 x2 ... xn]
             and None if there is no optimal solution.
    """
    tableau = _simplex_tableau(_slack_form(standard_form(lp)))

    while not _are_non_negative(tableau[0], 0, len(tableau)):
        pivot = _pivot_position(tableau)
        if not pivot:
            print("The give problem has no optimal solution.")
            return
        tableau[pivot[0]] = list(map(lambda x: x * (1 / tableau[pivot[0]][pivot[1]]), tableau[pivot[0]]))
        for i in [k for k in range(len(tableau)) if k != pivot[0]]:
            _add_row(tableau, -1 * (tableau[i][pivot[1]] / tableau[pivot[0]][pivot[1]]), pivot[0], i)

    solution = []  # top row; on the form [z x1 x2 ... xn s1 s2 ... sm b]
    columns = _columns(tableau)
    for column in columns:
        if _is_basic(column) >= 0:
            solution.append(tableau[_is_basic(column)][-1])
        else:
            solution.append(0)
    if lp.objective_function.objective is objective.minimize:  # set z = -z if we wanted minimum
        solution[0] = - solution[0]

    result = []  # pick out original variables by finding x' and x'' and summing them to get x = x' - x"
    i = 0
    real_index = 0
    decimal_places = 4
    while i < len(solution) - len(lp.constraints) - 1:
        if real_index in set([k + 1 for k in lp.free_variables]):
            result.append(round(solution[i] + solution[i + 1], decimal_places))
            i += 2
        else:
            result.append(round(solution[i], decimal_places))
            i += 1
        real_index += 1

    if display:
        output_string = ['solution: ', '\tZ = ' + str(result[0])]
        for i, value in enumerate(result[1:]):
            output_string.append('\tx_{} = '.format(i) + '%.3f' % value)
        print('\n'.join(output_string) + '\n')

    return result


def _columns(matrix: List[List[float]]) -> List[List[float]]:
    """
    Returns all the columns in a matrix.

    :param matrix: A matrix.
    :return: All the columns in the input matrix.
    """
    if not matrix:
        return []

    columns = []
    for j in range(len(matrix[0])):
        column = []
        for i in range(len(matrix)):
            column.append(matrix[i][j])
        columns.append(column)
    return columns


def _get_elements_in_column(matrix: List[List[float]], column: int, start: int, end: int) -> List[float]:
    """
    Returns the elements in a given column from entry start to entry end.

    :param matrix: A matrix.
    :param column: The index of the column.
    :param start: Index of the first element.
    :param end: Index of the last  element (exclusive).
    :return:
    """
    if not matrix:
        return []

    column_elements = []
    for i in range(start, end + 1):
        column_elements.append(matrix[i][column])
    return column_elements


def _is_basic(array: List[float]) -> int:
    """
    Checks whether an array is basic, that is has only one entry that is one
    and the rest zero. Returns the index of the 1, and -1 if it's non-basic.

    :param array: An array
    :return: If the array is basic, the index of the one will be returned, otherwise
             -1.
    """
    index = -1
    for i, entry in enumerate(array):
        if isinstance(entry, float):
            if not entry.is_integer():
                return -1
            else:
                entry = int(entry)
        if entry != 0 and entry != 1:
            return -1
        elif entry == 1:
            if index >= 0:
                return -1
            else:
                index = i
    return index


def _are_non_negative(array: List[float], start: int, end: int) -> bool:
    """
    Returns whether all elements in an array from index start to index end are non-negative.

    :param array: An array.
    :param start: The index of the start entry.
    :param end: The index of  the end entry.
    :return: True if all the elements within start and end in array are non-negative.
    """
    for i in range(start, end):
        if array[i] < 0:
            return False
    return True


def _pivot_position(tableau: List[List[float]]) -> (int, int):
    """
    Find the pivot position in the tableau. Returns None if there is no
    pivot position.

    :param tableau: A tableau.
    :return: The coordinates of the pivot entry if it exists, otherwise None.
    """
    if not tableau or not tableau[0]:
        return

    start = 1
    min_in_value_column = min(enumerate(_get_elements_in_column(tableau, len(tableau[0]) - 1, start, len(tableau) - 1)),
                              key=lambda x: x[1])
    if min_in_value_column[1] <= 0:
        row = min_in_value_column[0] + start
        column = min(enumerate(tableau[row][0:-1]), key=lambda x: x[1])[0]
        return row, column

    column = min(enumerate(tableau[0][0:-1]), key=lambda x: x[1])[0]
    row = -1
    minimum = math.inf
    for i in range(1, len(tableau)):
        if tableau[i][column] != 0:
            quotient = tableau[i][-1] / tableau[i][column]
            if minimum > quotient >= 0:
                minimum = quotient
                row = i
    return (row, column) if row >= 1 else None


def _add_row(matrix: List[List[float]], k: float, i: int, j: int) -> None:
    """
    Adds k times row i to row j in matrix in-place.

    :param matrix: Matrix to perform row-operations on.
    :param k: Multiple of row i to add to row j.
    :param i: The row to add to row j.
    :param j: the row to be added to.
    """
    matrix[j] = [a + k * b for a, b in zip(matrix[j], matrix[i])]


def _simplex_tableau(lp: linearProgram) -> List[List[float]]:
    """
    Returns a simplex tableau (or matrix) of the given linear program (in slack form):

    [1 -c^T 0]
    [0   A  b]

    where 1 is the objective function variable z and c are the coefficients in the objective function,
    A the matrix containing all of the coefficients in the constraints, and b the constants in
    right-hand-side of the constraints.

    :param lp: A linear program.
    :return: A simplex tableau for the linear program.
    """
    number_of_variables = len(lp.objective_function.coefficients)
    number_of_constraints = len(lp.constraints)
    tableau = [[0 for _ in range(number_of_variables + 2)] for _ in range(number_of_constraints + 1)]

    # first row [1 -c^T 0]
    tableau[0][0] = 1
    for i, coefficient in enumerate(lp.objective_function.coefficients):
        tableau[0][i + 1] = -1 * coefficient

    # following rows [0 A b]
    for i in range(1, len(tableau)):
        for j in range(1, len(tableau[i])):
            if j - 1 < number_of_variables:
                tableau[i][j] = lp.constraints[i - 1].coefficients[j - 1]
            else:
                tableau[i][j] = lp.constraints[i - 1].value
    return tableau


def standard_form(lp: linearProgram) -> linearProgram:
    """
    Creates a new linear programme equivalent to the input but
    in canonical form:
        1.  If the objective is to minimize, we turn into a problem of maximization through
            multiplication by -1.
        3.  If a constraint is an equality =, we change it into two inequalities <=, >=
        3.  Put constraints on correct form, that is c_1x_1 + ... + c_nx_n <= k constant.
        4.  Replace each free variable x with x = x' - x" where x', x" >= 0.

    :param lp: A linear program.
    :return: A linear programme equivalent to lp on standard form.
    """
    coefficients = _negate(lp.objective_function.coefficients) \
        if lp.objective_function.objective is objective.minimize \
        else lp.objective_function.coefficients[:]

    new_constraints = []
    for constr in lp.constraints:
        if constr.relation is inequality.leq:  # on the correct form
            new_constraints.append(constraint(constr.coefficients[:], inequality.leq, constr.value))
        elif constr.relation is inequality.geq:  # change >= to <= by multiplying through by -1
            new_constraints.append(constraint(_negate(constr.coefficients), inequality.leq, -1 * constr.value))
        elif constr.relation is inequality.equal:  # change = to one <= and one >= statement
            new_constraints.append(constraint(constr.coefficients[:], inequality.leq, constr.value))
            new_constraints.append(constraint(_negate(constr.coefficients), inequality.leq, -1 * constr.value))

    shift = 0
    for index in lp.free_variables:
        if index + 1 > len(coefficients):
            coefficients.append(-1 * coefficients[index + shift])  # take x to be x' and insert x" right after
        else:
            coefficients.insert(index + 1 + shift, -1 * coefficients[index + shift])
        for constr in new_constraints:
            if index + 1 > len(constr.coefficients):
                constr.coefficients.append(-1 * constr.coefficients[index + shift])
            else:
                constr.coefficients.insert(index + 1 + shift, -1 * constr.coefficients[index + shift])
        shift += 1

    return linearProgram(objectiveFunction(coefficients, objective.maximize), new_constraints)


def _slack_form(lp: linearProgram) -> linearProgram:
    """
    Returns a new linear program, after having added slack variables s_1, ..., s_n  to
    each constraint in a linear program on standard form.

    :param lp: A linear program on standard form
    :return: The inputs program on slack form.
    """
    number_of_variables = len(lp.objective_function.coefficients)
    number_of_slack_variables = len(lp.constraints)
    new_objective_function = objectiveFunction(
        lp.objective_function.coefficients + [0] * number_of_slack_variables,
        lp.objective_function.objective
    )
    new_constraints = []
    for i, constr in enumerate(lp.constraints):
        new_coefficients = constr.coefficients + [0] * number_of_slack_variables
        new_coefficients[number_of_variables + i] = 1
        new_constraints.append(constraint(
            new_coefficients,
            inequality.equal,
            constr.value
        ))
    return linearProgram(new_objective_function, new_constraints)


def _negate(array: List[float]) -> List[float]:
    """
    Returns a new array in which all every number in the input array has been
    multiplied by -1.

    :param array: Input list.
    :return: New list with every element in the input list multiplied by -1.
    """
    return [-1 * k for k in array]
