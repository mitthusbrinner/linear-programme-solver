import unittest
import lp as lp


class TestLp(unittest.TestCase):

    def test_negate(self):
        array = []
        self.assertEqual(array, lp._negate(array))

        array = [0] * 1000000
        self.assertEqual(array, lp._negate(array))

        array = [-1, -1, -1]
        negated_array = [1, 1, 1]
        self.assertEqual(negated_array, lp._negate(array))

        array = [1] * 1000000
        negated_array = [-1] * 1000000
        self.assertEqual(negated_array, lp._negate(array))

    def test_add_row(self):
        matrix = [
            [0, 0, 0],
            [1, 1, 1],
            [0, 0, 0]
        ]
        lp._add_row(matrix, 0, 0, 0)
        self.assertEqual([
            [0, 0, 0],
            [1, 1, 1],
            [0, 0, 0]
        ], matrix)

        lp._add_row(matrix, 5.2, 1, 0)
        self.assertEqual([
            [5.2, 5.2, 5.2],
            [1, 1, 1],
            [0, 0, 0]
        ], matrix)

    def test_are_non_negative(self):
        self.assertEqual(True, lp._are_non_negative([0] * 1000000, 0, 1000000))
        self.assertEqual(True, lp._are_non_negative([100] * 1000000, 0, 1000000))
        self.assertEqual(True, lp._are_non_negative([1, 2, 3, -1, 0, -1, 2], 0, 3))
        self.assertEqual(False, lp._are_non_negative([1, 2, 3, -1, 0, -1, 2], 0, 7))
        self.assertEqual(False, lp._are_non_negative([-1] * 100000, 0, 100000))

    def test_is_basic(self):
        self.assertEqual(1, lp._is_basic([0, 1, 0, 0, 0, 0, 0]))
        self.assertEqual(1, lp._is_basic([0, 1.0000000, 0, 0, 0, 0, 0]))
        self.assertEqual(-1, lp._is_basic([0, 1.0, 0.1111, 0, 0, 0, 0]))
        self.assertEqual(0, lp._is_basic([1]))
        self.assertEqual(-1, lp._is_basic([0]))

    def test_columns(self):
        self.assertEqual([], lp._columns([]))
        self.assertEqual([[1]], lp._columns([[1]]))

        matrix = [
            [0,   1,  2,  3,  4,  5],
            [9,  10, 11, 12, 13, 14],
            [18, 19, 20, 21, 22, 23]
        ]
        self.assertEqual([
            [0, 9, 18],
            [1, 10, 19],
            [2, 11, 20],
            [3, 12, 21],
            [4, 13, 22],
            [5, 14, 23]
        ], lp._columns(matrix))

    def test_get_elements_in_column(self):
        self.assertEqual([], lp._get_elements_in_column([], 0, 0, 0))
        matrix = [
            [0,   1,  2,  3,  4, 5],
            [9,  10, 11, 12, 13, 14],
            [18, 19, 20, 21, 22, 23]
        ]
        self.assertEqual([2], lp._get_elements_in_column(matrix, 2, 0, 0))
        self.assertEqual([0, 9, 18], lp._get_elements_in_column(matrix, 0, 0, 2))
        self.assertEqual([9, 18], lp._get_elements_in_column(matrix, 0, 1, 2))
        self.assertEqual([4, 13], lp._get_elements_in_column(matrix, 4, 0, 1))

    def test_objectiveFunction(self):
        self.assertRaises(Exception, lp.objectiveFunction.__init__, [], lp.objective.maximize)
        self.assertRaises(Exception, lp.objectiveFunction.__init__, [1], None)

    def test_linearProgram(self):
        obj_func = lp.objectiveFunction([1, 2, 3], lp.objective.maximize)
        constr1 = lp.constraint([1, 2], lp.inequality.leq, 3)
        constr2 = lp.constraint([1, 2, 4], lp.inequality.leq, 3)
        self.assertRaises(Exception, lp.linearProgram.__init__, obj_func, [constr1, constr2])

    def _test_simplex_tableau(self):
        linear_prog = lp.linearProgram(
            lp.objectiveFunction([2.5, 3], lp.objective.maximize),
            [
                lp.constraint([3, 6], lp.inequality.leq, 90),
                lp.constraint([2, 1], lp.inequality.leq, 35),
                lp.constraint([1, 1], lp.inequality.leq, 20)
            ],
            free_variables=[]
        )
        self.assertEqual([
            [1, -2.5, -3, 0, 0, 0, 0],
            [0,    3,  6, 1, 0, 0, 90],
            [0,    2,  1, 0, 1, 0, 35],
            [0,    1,  1, 0, 0, 1, 20]
        ], lp._simplex_tableau(linear_prog))

    def test_pivot_position(self):
        self.assertEqual(None, lp._pivot_position([[]]))

        tableau1 = [
            [1, -40, -30, 0, 0, 0],
            [0, 1, 1, 1, 0, 5],
            [0, 2, -3, 0, 1, -12]
        ]
        self.assertEqual((2, 2), lp._pivot_position(tableau1))

        tableau2 = [
            [1, -2.5, -3, 0, 0, 0, 0],
            [0, 3, 6, 1, 0, 0, 90],
            [0, 2, 1, 0, 1, 0, 35],
            [0, 1, 1, 0, 0, 1, 20]
        ]
        self.assertEqual((1, 2), lp._pivot_position(tableau2))

    def test_standard_form(self):
        linear_prog1 = lp.linearProgram(
            lp.objectiveFunction([40, 30], lp.objective.minimize),
            [
                lp.constraint([1, 1], lp.inequality.leq, 5),
                lp.constraint([-2, 3], lp.inequality.geq, 12),
            ],
            free_variables=[]
        )
        standard_form = lp.standard_form(linear_prog1)
        self.assertEqual(lp.objective.maximize, standard_form.objective_function.objective)
        self.assertEqual([1, 1], standard_form.constraints[0].coefficients)
        self.assertEqual(lp.inequality.leq, standard_form.constraints[0].relation)
        self.assertEqual(5, standard_form.constraints[0].value)

        self.assertEqual([2, -3], standard_form.constraints[1].coefficients)
        self.assertEqual(lp.inequality.leq, standard_form.constraints[1].relation)
        self.assertEqual(-12, standard_form.constraints[1].value)

    def test_slack_form(self):
        linear_prog = lp.linearProgram(
            lp.objectiveFunction([2.5, 3], lp.objective.maximize),
            [
                lp.constraint([3, 6], lp.inequality.leq, 90),
                lp.constraint([2, 1], lp.inequality.leq, 35),
                lp.constraint([1, 1], lp.inequality.leq, 20)
            ],
            free_variables=[]
        )
        slack_form = lp._slack_form(linear_prog)
        self.assertEqual([2.5, 3, 0, 0, 0], slack_form.objective_function.coefficients)
        self.assertEqual([3, 6, 1, 0, 0], slack_form.constraints[0].coefficients)
        self.assertEqual([2, 1, 0, 1, 0], slack_form.constraints[1].coefficients)
        self.assertEqual([1, 1, 0, 0, 1], slack_form.constraints[2].coefficients)

    def test_solve(self):
        # non-standard
        linear_prog1 = lp.linearProgram(
            lp.objectiveFunction([40, 30], lp.objective.maximize),
            [
                lp.constraint([1, 1], lp.inequality.leq, 5),
                lp.constraint([-2, 3], lp.inequality.geq, 12),
            ],
            free_variables=[]
        )
        self.assertEqual([156, 3.0/5.0, 22.0/5.0], lp.solve(linear_prog1))

        linear_prog2 = lp.linearProgram(
            lp.objectiveFunction([-1, 2], lp.objective.minimize),
            [
                lp.constraint([2, 3], lp.inequality.leq, 6),
                lp.constraint([2, 1], lp.inequality.leq, 14),
            ],
            free_variables=[]
        )
        self.assertEqual([-3, 3, 0], lp.solve(linear_prog2))

        # standard
        linear_prog3 = lp.linearProgram(
            lp.objectiveFunction([2.5, 3], lp.objective.maximize),
            [
                lp.constraint([3, 6], lp.inequality.leq, 90),
                lp.constraint([2, 1], lp.inequality.leq, 35),
                lp.constraint([1, 1], lp.inequality.leq, 20)
            ],
            free_variables=[]
        )
        self.assertEqual([55, 10, 10], lp.solve(linear_prog3))
