# Fassio's Genetic Programming Toolbox

This library is developed for the course Computational Intelligence at Politecnico di Torino.

The inner working of this library is influenced by Operon, a C++ framework developed by Heuristic and Evolutionary Algorithms Laboratory at University of Applied Sciences Upper Austria.

## Python Version

To maximize the parallelization capabilities of this package, we recommend using `Python 3.13 with free-threading support`.

## Usage

```example.py```

## TODO

A list of things that maybe (denoted by (?)) should be done, but I don't fully know how or which to do:

- [X] Understand and implement techniques to limit bloat (Note: an individual is "fat" when it exceeds max_depth or max_length). Like:
  - (DONE) improve the implementation of create_grow to always stay inside the limits
  - implement genetic operators that cannot create a fat individual (Operon)
  - (DONE) reject fat individual from offspring population (Operon)
  - be more aggressive with the probability of hoist mutation
- [ ] Implement some niching strategy to favor diversity
- [ ] Implement early stopping conditions
- [ ] (TOO ADVANCED) try to enforce smoothness in the train domain (all correct values (no NaN) and smooth derivative)
- [ ] (TOO ADVANCED) constant values optimization with gradient descent
- [X] (NOT A PROBLEM) Understand how to deal with invalid trees, that return ```np.nan```, in the first generation, because they lower the initial genetic diversity (?)
- [ ] Optimize tree ```__call__``` to cache results (?)
- [X] Implement genetic operators: change of constants
- [ ] Implement other genetic operators (?)
- [ ] Implement other numpy operators (?)

## Ideas from the Prof

- collect statistics of the success of the genop
- collect statistics on diversity
- self-adaptation of the usefulness of the genop
- constant evaluation to reduce bloat
- check for bugs

## License

MIT License

## Author

Davide Fassio
