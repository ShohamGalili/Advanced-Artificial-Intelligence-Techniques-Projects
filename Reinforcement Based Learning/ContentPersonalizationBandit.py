# Hadas Yossefzada ID: 213486764
# Shoham Galili ID: 208010785

import numpy as np
import time
from bandits import Bandit


class ContentPersonalizationBandit(Bandit):
    def __init__(self, n, probas=None):
        """
        Initializes the ContentPersonalizationBandit object.

        Parameters:
        - n (int): Number of content categories.
        - probas (list, optional): Probabilities of engagement for each content category.
                                   If None, probabilities are generated randomly.
        """
        assert probas is None or len(probas) == n
        self.n = n # Number of content categories
        # Set the bandits' rewards randomly
        if probas is None:
            np.random.seed(int(time.time()))
            self.probas = [np.random.random() for _ in range(self.n)]
        else:
            self.probas = probas

        # Maximal reward
        self.best_proba = max(self.probas)

    def generate_reward(self, i):
        """
        Simulates the generation of a reward for selecting the i-th content category.

        Parameters:
        - i (int): Index of the selected content category.

        Returns:
        - reward (int): Reward generated.
                        1 if the user engages with the content,
                        0 if the user does not engage with the content.
        """
        # Check if the user clicks on the i-th ad placement based on its probability
        if np.random.random() < self.probas[i]:
            return 1  # User clicks on the ad
        else:
            return 0  # User does not click on the ad