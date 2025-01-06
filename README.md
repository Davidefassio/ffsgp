# Fassio's Genetic Programming Toolbox

This library is developed for the course Computational Intelligence at Politecnico di Torino.

The inner working of this library is influenced by Operon, a C++ framework developed by Heuristic and Evolutionary Algorithms Laboratory at University of Applied Sciences Upper Austria.

## Python Version

To maximize the parallelization capabilities of this package, we recommend using `Python 3.13 with free-threading support`.

## Usage

Examples demonstrating the training process and the underlying data structures are provided in `example.py`.

The script `main.py` contains the implementation for fitting all the formulas to the dataset.

## Implementation Details

This section provides an overview of key implementation details of the library, presented in no particular order, to offer clarity and transparency.

The library's architecture is heavily inspired by Operon, known for its exceptional speed and parallelizability.

### Node

The Node is a fundamental component of the architecture. To store nodes efficiently in a `np.array` in-place (without pointers or redirections), they must have a fixed size and structure. This is achieved by using six 64-bit fields:

- `f` (integer): Represents the operation (`+`, `-`, `*`, `/`, ...)
- `name` (integer): Denotes the variable (`x`, `y`, `z`, ...)
- `value` (float): Stores the value of a constant
- `arity` (integer): Indicates the arity of the node:
  - `1` for variables and constants
  - `1` or `2` for operators
- `length` (integer): Indicates the length of the subtree rooted at this node
- `depth` (integer): Indicates the depth of the subtree rooted at this node

All unused fields are set to `-1`.

### Tree

A Tree is represented as a `np.array` of nodes in postfix order.

The tree structure provides several utilities, including:

- Evaluation: Efficiently computes the value of the formula represented by the tree for multiples data points using the vectorization offered by `numpy`.
- Metric Updates: Calculates and updates properties such as `length` and `depth` for each node.
- Human-Readable and Python Executable Representation: Converts the tree into a format that is easy to interpret and execute.

### Genetic Operators

The following genetic operators are implemented to evolve solutions:

- Crossover Subtree: Swap a subtree in one parent with a subtree from another parent.
- Mutation Single Point: Modifies a single node in the tree while maintaing the type.
- Mutation Constants: Adjusts the values of all constants to refine solutions.
- Mutation Hoist: Replaces a tree with one of its subtrees, effectively reducing tree size.
- Mutation Subtree: Replaces a subtree with a new subtree generated randomly with the grow method.

These operators may exceed predefined depth and length limits, allowing exploration of diverse solution spaces.

### Creation of the Initial Population

The library offers multiple strategies for creating the initial population of trees:

- Full: Generates trees where all nodes at each level are fully populated with operators or terminals until the maximum depth.
- Grow: Randomly selects operators and terminals at each level, producing trees with varying structures.
- Half and Half: Combines the Full and Grow methods to generate a diverse population.
- Ramped Half and Half: Randomly choosed the Full or Grow method to generate the next subtree, but if it is already present in the population is discarded.

### Parent Selection

Parent selection is performed using tournament selection with fitness hole, a method that selects candidates based on their fitness but introduces controlled randomness to reduce bloat.

### Metrics

The following metrics are implemented to evaluate solutions:

- Mean Absolute Error (MAE): Measures the average absolute difference between predicted and actual values.
- Mean Squared Error (MSE): Calculates the average squared difference between predicted and actual values.
- Root Mean Squared Error (RMSE): Computes the square root of the MSE for interpretability in the original scale.
- Normalized Mean Squared Error (NMSE): Normalizes the MSE relative to the variance of the target values.
- R² (Coefficient of Determination): Evaluates how well the model explains the variance in the target data.

### Trainer

The Trainer orchestrates the entire training process. It accepts input data points and produces the best formula discovered through evolution.

Key features include:

- Generational Approach: The evolutionary process proceeds in generations, employing elitism to retain the best individuals and a Hall of Fame to track top solutions.
- `varAnd` Method: Applies crossover to generate offspring and performs mutations with a configurable probability. Offspring that violate depth or length constraints are discarded. This step is parallelized for maximum performance.
- Data Normalization: Normalizes target values (`y`) to have a mean of 0 and standard deviation of 1, improving convergence. After training, a denormalization step is appended to the final formula to restore the original value range.

## License

MIT License

## Author

Davide Fassio
