import ffsgp as ffs

import numpy as np
from functools import partial


def example1():
    """
    Train a population of trees to match a given formula!
    """
    # Choose which formula to emulate
    formula = ffs.Tree(2, ffs.Var(0), ffs.mul, 1/3, ffs.sub)
    # formula = ffs.Tree(ffs.Var(0), ffs.abs, ffs.sin)
    # formula = ffs.Tree(ffs.Var(0), 100, ffs.add, 0.5, ffs.pow, ffs.cos, ffs.log)

    x = np.linspace(-10, 10, 101).reshape(1, -1)
    y = formula(x)

    # Note:
    #  initial_population wants a population, not a function to create one
    #  fitness is maximized! So for mae, mse, nmse, rmse use a lambda to negate them
    #  functools.partial is used to fix some parameters (used for parent_selection and mutations)
    #  n_jobs is the number of threads to run parallely. We use scikit-learn convention (look: utility_threads.py)
    trainer = ffs.Trainer(x=x, y=y,
                          normalize=False,
                          initial_population=ffs.init_pop_ramp(1000, 1, 6, 20, 0.2),
                          offspring_mult=4,
                          n_generations=50,
                          fitness=lambda yt, yp: -ffs.mse(yt, yp),
                          parent_selection=partial(ffs.tournament, n=3, p_fh=0.2),
                          crossover=ffs.crossover_subtree,
                          mutations=[partial(ffs.mutation_single_point, nvars=1), ffs.mutation_hoist],
                          probs_mutation=[0.1, 0.01],
                          elitism=0.1,
                          verbose=True,
                          n_jobs=-1)

    ft = trainer.train()

    print(ft.to_human())
    print(ft.fitness)


def example2():
    """
    Create two trees, generate two offspring and plot the results!
    """

    import matplotlib.pyplot as plt  # When imported, no more parallelism!

    # Data
    x = np.linspace(-5, 5, 101)
    y = np.linspace(-5, 5, 101)
    X, Y = np.meshgrid(x, y)

    t1 = ffs.Tree(ffs.Var(0), ffs.Var(0), ffs.mul, ffs.Var(1), 2,
                  ffs.pow, ffs.add, 1 / 2, ffs.pow, ffs.sin)
    t2 = ffs.Tree(2, ffs.Var(0), ffs.mul, 1/3, ffs.sub)

    t3, t4 = ffs.crossover_subtree(t1, t2)

    Z1 = t1(np.vstack((X.reshape(-1), Y.reshape(-1)))).reshape(X.shape)
    Z2 = t2(np.vstack((X.reshape(-1), Y.reshape(-1)))).reshape(X.shape)
    Z3 = t3(np.vstack((X.reshape(-1), Y.reshape(-1)))).reshape(X.shape)
    Z4 = t4(np.vstack((X.reshape(-1), Y.reshape(-1)))).reshape(X.shape)

    # Create the plot
    fig = plt.figure(figsize=(10, 8))

    ax1 = fig.add_subplot(2, 2, 1, projection='3d')
    ax1.plot_surface(X, Y, Z1, cmap='viridis', edgecolor='none')
    ax1.set_title("Parent 1")
    ax1.set_xlabel('x', fontsize=14)
    ax1.set_ylabel('y', fontsize=14)
    ax1.set_zlabel('z', fontsize=14)

    ax2 = fig.add_subplot(2, 2, 2, projection='3d')
    ax2.plot_surface(X, Y, Z2, cmap='viridis', edgecolor='none')
    ax2.set_title("Parent 2")
    ax2.set_xlabel('x', fontsize=14)
    ax2.set_ylabel('y', fontsize=14)
    ax2.set_zlabel('z', fontsize=14)

    ax3 = fig.add_subplot(2, 2, 3, projection='3d')
    ax3.plot_surface(X, Y, Z3, cmap='viridis', edgecolor='none')
    ax3.set_title("Offspring 1")
    ax3.set_xlabel('x', fontsize=14)
    ax3.set_ylabel('y', fontsize=14)
    ax3.set_zlabel('z', fontsize=14)

    ax4 = fig.add_subplot(2, 2, 4, projection='3d')
    ax4.plot_surface(X, Y, Z4, cmap='viridis', edgecolor='none')
    ax4.set_title("Offspring 2")
    ax4.set_xlabel('x', fontsize=14)
    ax4.set_ylabel('y', fontsize=14)
    ax4.set_zlabel('z', fontsize=14)

    # Show plot
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    example1()
    example2()
