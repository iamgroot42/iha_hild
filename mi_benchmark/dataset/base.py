import numpy as np


class Dataset:
    def __init__(self, name, train, test, num_classes: int):
        self.name = name
        self.train = train
        self.test = test
        self.num_classes = num_classes
    
    def get_train_data(self):
        return self.train
    
    def get_test_data(self):
        return self.test
    
    def get_data(self):
        return self.get_train_data(), self.get_test_data()

    def make_splits(
        self, seed: int, pkeep: float, exp_id: int = None, num_experiments: int = None
    ):
        """
        For random split generation, follow same setup as Carlini/Shokri.
        This is the function to generate subsets of the data for training models.

        First, we get the training dataset.

        Then, we compute the subset. This works in one of two ways.

        1. If we have a seed, then we just randomly choose examples based on
           a prng with that seed, keeping pkeep fraction of the data.

        2. Otherwise, if we have an experiment ID, then we do something fancier.
           If we run each experiment independently then even after a lot of trials
           there will still probably be some examples that were always included
           or always excluded. So instead, with experiment IDs, we guarantee that
           after num_experiments are done, each example is seen exactly half
           of the time in train, and half of the time not in train.

        """

        def make_split(data, sd):
            if num_experiments is not None:
                np.random.seed(sd)
                keep = np.random.uniform(0, 1, size=(num_experiments, len(data.data)))
                order = keep.argsort(0)
                keep = order < int(pkeep * num_experiments)
                keep = np.array(keep[exp_id], dtype=bool)
            else:
                np.random.seed(sd)
                keep = np.random.uniform(0, 1, size=len(data.data)) <= pkeep
            # Create indices corresponding to keep
            indices = np.arange(len(data.data))
            return indices[keep]

        # Split train,test such that every datapoint seen with probability pkeep
        train_keep = make_split(self.train, seed)
        test_keep = make_split(self.test, seed + 1)

        return train_keep, test_keep, self.train, self.test
