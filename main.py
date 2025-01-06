import ffsgp as ffs

import numpy as np
from functools import partial


def main():
    """
    Train a population of trees to match a dataset!
    """
    list_of_goats = []

    # Load training dataset
    for i in range(1, 9):
        dataset = np.load(f"data/problem_{i}.npz")

        # Extract data and number of variables
        x = dataset['x']
        y = dataset['y']

        nvars = x.shape[0]

        champions = []
        for it in range(50):
            print(f"# P{i}: {it} ###")

            trainer = ffs.Trainer(x=x, y=y,
                                normalize=True,
                                initial_population=ffs.init_pop_ramp(1000, nvars, 6, 20, 0.2),
                                offspring_mult=4,
                                n_generations=50,
                                fitness=lambda yt, yp: ffs.r2(yt, yp),
                                parent_selection=partial(ffs.tournament, n=2, p_fh=0.1),
                                crossover=ffs.crossover_subtree,
                                mutations=[partial(ffs.mutation_single_point, nvars=nvars), ffs.mutation_hoist, ffs.mutation_constant],
                                probs_mutation=[0.1, 0.01, 0.5],
                                max_depth=20,
                                max_length=50,
                                elitism=0.1,
                                verbose=False,
                                n_jobs=-1)

            ft = trainer.train()
            champions.append(ft)

        goats = sorted(champions, key=lambda t: t.fitness, reverse=True)
        list_of_goats.append(goats)

    for n, log in enumerate(list_of_goats):
        print(f"Problem {n + 1}")
        for g in log[:10]:
            print(f"{g.fitness}: {g.to_human()}")
        print()


if __name__ == '__main__':
    main()
