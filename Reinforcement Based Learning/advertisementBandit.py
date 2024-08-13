# Hadas Yossefzada ID:213486764
# Shoham Galili ID: 208010785

import numpy as np
import time
from bandits import Bandit

class AdvertisementBandit(Bandit):
    def __init__(self, n, probas=None):
        """
        Initializes the AdvertisementBandit object.

        Parameters:
            n (int): Number of ad placements.
            probas (list, optional): List of probabilities for each ad placement.
                                      If not provided, probabilities are generated randomly.
        """
        # Check if probas list is provided and has correct length
        assert probas is None or len(probas) == n

        self.n = n
        # Set the bandits' rewards randomly
        if probas is None:
            np.random.seed(int(time.time()))
            self.probas = [np.random.random() for _ in range(self.n)]
        else:
            self.probas = probas

        # Maximum probability among all ad placements
        self.best_proba = max(self.probas)

    def generate_reward(self, i):
        """
        Simulates the generation of a reward for selecting the i-th ad placement.

        Parameters:
        - i (int): The index of the ad placement.

        Returns:
        - reward (int): The reward generated.
                        1 if the user clicks on the ad,
                        0 if the user does not click on the ad.
        """
        # Check if the user clicks on the i-th ad placement based on its probability
        if np.random.random() < self.probas[i]:
            return 1  # User clicks on the ad
        else:
            return 0  # User does not click on the ad