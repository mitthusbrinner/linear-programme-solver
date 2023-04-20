# LP Linear programming 

This library implements functions for solving bounded [linear programmes (LP)](https://en.wikipedia.org/wiki/Linear_programming) and for putting it on [standard form](https://en.wikipedia.org/wiki/Linear_programming#Standard_form).

The algorithm used for solving the LP:s is the [Simplex Algorithm](https://en.wikipedia.org/wiki/Simplex_algorithm), which in this implementation automatically puts the input problem on standard form before solving. 

[<img src="https://ds055uzetaobb.cloudfront.net/brioche/uploads/3mbYZ5LMun-linear-programming-proof.png?width=3000" width=400>]()

### Roadmap

* The API of this library is frozen.
* Version number adheres to [semantic versioning](https://semver.org/).

### Documentation 
#### Types
```python
    class objective(Enum):
        """ 
        The objective in the objective function; either maximize or minimize
        """
    
    class inequality(Enum):
        """ 
        The relationship in an inequality, either geq (>=), equal or leq (<=)
        """
        
    class constraint:
        """ 
        Represents a constraint by a coefficient array, an inequality and a constant.
        
        Example: 4x_1 + 6x_3 <= 3 gets represented by [4, 0, 6], inequality.leq and 3.
        """
        
    class objectiveFunction:
        """
        Represents an objective function by a coefficient array and an objective.
        
        Example: Maximize 3x_1 + 4x_2 - 34x_3 gets represented by [3, 4, -32] 
        and objective.maximize
        """
        
    class linearProgram:
        """
        Represents a linear program by a objective function and a list of constraints.
        """
```
#### Functions 
````python
    def standard_form(lp: linearProgram) -> linearProgram:
	    """
	    Creates a new linear program equivalent to the input but
	    in canonical form.

	    :param lp: A linear program.
	    :return: A linear program equivalent to lp on standard form.
	    """

    def solve(lp: linearProgram, display=False) -> List[float]:
	    """
	    Solves a linear program using the Simplex Algorithm.

	    :param lp: a linear program.
	    :param display: print the solution in a human-readable way if set to True else not.
	    :return: solution to the linear program on the form  [Z x1 x2 ... xn]
		     and None if there is no optimal solution.
	    """
````
