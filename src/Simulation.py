ALGORITHMS = {
    'name': lambda: 0
}


def main():
    columns = ['trial', 'time-step', 'opinion', 'activated', 'algorithm', 'seed-size']
    model = BIC_Model()

    for trial in tqdm(range(n_trials)):
        for n_seeds in seed_set_sizes:
            for algorithm in algorithms:
                ...
                for time_step in range(time_horizon):
                    ...
