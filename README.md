# Playing with functional dependencies

In the [functional_deps.ipynb](https://github.com/pratik2358/fucntional_dep/blob/main/functional_deps.ipynb) file, run the first cell as it is.

You'll find the example usage in the subsequent cells.

The functional dependencies are expressed as a list of tuples. The first set in the each tuple represents the left hand side of the dependency and the second set in each tuple represents the right hand side of the dependency.

```
attributes = {'A', 'B', 'C', 'D', 'E'}
fds = [
    ({'A'}, {'A', 'B', 'C'}),
    ({'A', 'B'}, {'A'}),
    ({'B', 'C'}, {'A', 'D'}),
    ({'B'}, {'A', 'B'}),
    ({'C'}, {'D'})
]
```
The above expression can be translated as below:

R = {A, B, C, D, E}

$\Sigma$ = {{A} -> {A, B, C},\
{A, B} -> {A},\
{B, C} -> {A, D},\
{B} -> {A, B},\
{C} -> {D}}

# Finding functional dependencies from a pandas dataframe

In the [discover.ipynb](https://github.com/pratik2358/fucntional_dep/blob/main/discover.ipynb) file, run the first cell as it is. The next cells show two small examples to find functional dependency from a given relation (table).

In this, if you have a table (pandas dataframe), you can use the algorithm to find the set of functional dependencies that can be inferred from it.
