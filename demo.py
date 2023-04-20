import lp as lp


def main() -> None:
    my_objective_function = lp.objectiveFunction([40, 30], lp.objective.maximize)

    my_constraint1 = lp.constraint([1, 1], lp.inequality.leq, 5)
    my_constraint2 = lp.constraint([-2, 3], lp.inequality.geq, 12)

    my_lp = lp.linearProgram(my_objective_function,
                             [my_constraint1, my_constraint2],
                             free_variables=[])

    print(my_lp)
    print("On standard form: \n" +
          str(lp.standard_form(my_lp)))

    solution = lp.solve(my_lp, display=True)


if __name__ == '__main__':
    main()
