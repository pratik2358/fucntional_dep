# Playing with functional dependencies

In the functional_deps.ipynb file, run the first cell as it is.

You'll find the example usage in the subsequent cells.

The functional dependencies are expressed as a list of tuples. The first set in the each tuple represent the left hand side of the dependency and the second set in each tuple represent the right hand side of the dependency.

```
fds = [
    ({'A'}, {'A', 'B', 'C'}),
    ({'A', 'B'}, {'A'}),
    ({'B', 'C'}, {'A', 'D'}),
    ({'B'}, {'A', 'B'}),
    ({'C'}, {'D'})
]
```
The above expression can be translated as below:

{{A} -> {A, B, C}},
{A, B} -> {A},
{B, C} -> {A, D},
{B} -> {A, B},
{C} -> {D}}
