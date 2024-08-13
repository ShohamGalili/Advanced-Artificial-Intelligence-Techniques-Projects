# Hadas Yossefzada ID: 213486764
# Shoham Galili ID: 208010785

from __future__ import division
from ContentPersonalizationBandit import ContentPersonalizationBandit
import numpy as np
import time

class Solver(object):

    def __init__(self, bandit):
        """
        Initializes the Solver object.

        Parameters:
        - bandit (ContentPersonalizationBandit): The target bandit to solve.
        """
        assert isinstance(bandit, ContentPersonalizationBandit)
        np.random.seed(int(time.time()))

        self.bandit = bandit

        self.counts = [0] * self.bandit.n
        self.actions = []  # A list of machine ids, 0 to bandit.n-1.
        self.regret = 0.  # Cumulative regret.
        self.regrets = [0.]  # History of cumulative regret.

        self.Name = []

    # Compute loss due to non-optimal policy
    def update_regret(self, i):
        """
        Updates the cumulative regret.

        Parameters:
        - i (int): Index of the selected machine.
        """
        # self.bandit.best_proba - maximal reward
        self.regret += self.bandit.best_proba - self.bandit.probas[i]
        self.regrets.append(self.regret)

    @property
    def estimated_probas(self):
        """
        Estimates the probabilities of each action.

        Returns:
        - estimated_probas (list): Estimated probabilities of each action.
        """
        raise NotImplementedError

    def run_one_step(self):
        """
        Runs one step of the solver.

        Returns:
        - i (int): Index of the selected machine to take action on.
        """
        raise NotImplementedError

    def run(self, num_steps):
        """
        Runs the solver for a given number of steps.

        Parameters:
        - num_steps (int): Number of steps to run the solver.
        """
        assert self.bandit is not None

        for k in range(num_steps):
            i = self.run_one_step()

            self.counts[i] += 1

            # Store results for statistics
            self.actions.append(i)
            self.update_regret(i)


class EpsilonGreedy(Solver):

    def __init__(self, bandit, eps, init_proba=1.0):
        """
        Initializes the EpsilonGreedy solver.

        Parameters:
        - bandit (ContentPersonalizationBandit): The target bandit to solve.
        - eps (float): The probability to explore at each time step.
        - init_proba (float): Default to be 1.0; optimistic initialization.
        """
        super(EpsilonGreedy, self).__init__(bandit)

        self.Name = 'EpsilonGreedy'

        assert 0. <= eps <= 1.0
        self.eps = eps

        # Optimistic initialization
        self.estimates = [init_proba] * self.bandit.n

    @property
    def estimated_probas(self):
        """
        Estimates the probabilities of each action.

        Returns:
        - estimated_probas (list): Estimated probabilities of each action.
        """
        return self.estimates

    def run_one_step(self):
        """
        Runs one step of the EpsilonGreedy solver.

        Returns:
        - i (int): Index of the selected machine to take action on.
        """
        if np.random.random() < self.eps:
            # Let's do random exploration!
            i = np.random.randint(0, self.bandit.n)
        else:
            # Pick the best one.
            max_reward = -np.inf
            for k in range(self.bandit.n):
                CurrEstimate = self.estimates[k]

                if CurrEstimate > max_reward:
                    max_reward = CurrEstimate
                    i = k

        # Connect to environment
        r = self.bandit.generate_reward(i)

        # Update estimate of mean
        self.estimates[i] = self.estimates[i] + 1. / (self.counts[i] + 1) * (r - self.estimates[i])

        return i


class UCB1(Solver):

    def __init__(self, bandit, init_proba=1.0):
        """
        Initializes the UCB1 solver.

        Parameters:
        - bandit (ContentPersonalizationBandit): The target bandit to solve.
        - init_proba (float): Default to be 1.0; optimistic initialization.
        """
        super(UCB1, self).__init__(bandit)

        self.Name = 'UCB1'

        self.t = 0
        self.estimates = [init_proba] * self.bandit.n

    @property
    def estimated_probas(self):
        """
        Estimates the probabilities of each action.

        Returns:
        - estimated_probas (list): Estimated probabilities of each action.
        """
        return self.estimates

    def run_one_step(self):
        """
        Runs one step of the UCB1 solver.

        Returns:
        - i (int): Index of the selected machine to take action on.
        """
        self.t += 1

        # Pick the best one with consideration of upper confidence bounds.
        max_reward = -np.inf
        for k in range(self.bandit.n):
            CurrEstimate = self.estimates[k] + np.sqrt(2 * np.log(self.t) / (1 + self.counts[k]))

            if CurrEstimate > max_reward:
                max_reward = CurrEstimate
                i = k

        r = self.bandit.generate_reward(i)

        self.estimates[i] += 1. / (self.counts[i] + 1) * (r - self.estimates[i])

        return i


class BayesianUCB(Solver):

    def __init__(self, bandit, c=3, init_a=1, init_b=1, init_proba=1.0):
        """
        Initializes the BayesianUCB solver.

        Parameters:
        - bandit (ContentPersonalizationBandit): The target bandit to solve.
        - c (float): How many standard deviations to consider as upper confidence bound.
        - init_a (int): Initial value of alpha in Beta(alpha, beta).
        - init_b (int): Initial value of beta in Beta(alpha, beta).
        """
        super(BayesianUCB, self).__init__(bandit)

        self.Name = 'BayesianUCB'

        self.c = c
        self._as = [init_a] * self.bandit.n
        self._bs = [init_b] * self.bandit.n

        self.Xestimates = [init_proba] * self.bandit.n
        self.X2estimates = [init_proba] * self.bandit.n

    @property
    def estimated_probas(self):
        """
        Estimates the probabilities of each action.

        Returns:
        - estimated_probas (list): Estimated probabilities of each action.
        """
        return [self._as[i] / float(self._as[i] + self._bs[i]) for i in range(self.bandit.n)]

    def run_one_step(self):
        """
        Runs one step of the BayesianUCB solver.

        Returns:
        - i (int): Index of the selected machine to take action on.
        """
        # Pick the best one with consideration of upper confidence bounds.
        max_reward = -np.inf
        for k in range(self.bandit.n):
            CurrEstimate = self.Xestimates[k] + 2 * np.sqrt(self.X2estimates[k])

            if CurrEstimate > max_reward:
                max_reward = CurrEstimate
                i = k

        r = self.bandit.generate_reward(i)

        # Update Gaussian posterior
        self._as[i] += r
        self._bs[i] += (1 - r)

        self.Xestimates[i] += 1. / (self.counts[i] + 1) * (r - self.Xestimates[i])
        self.X2estimates[i] += 1. / (self.counts[i] + 1) * (r ** 2 - self.X2estimates[i])

        return i


class ThompsonSampling(Solver):

    def __init__(self, bandit, init_a=1, init_b=1):
        """
        Initializes the ThompsonSampling solver.

        Parameters:
        - bandit (ContentPersonalizationBandit): The target bandit to solve.
        - init_a (int): Initial value of alpha in Beta(alpha, beta).
        - init_b (int): Initial value of beta in Beta(alpha, beta).
        """
        super(ThompsonSampling, self).__init__(bandit)

        self.Name = 'ThompsonSampling'

        self._as = [init_a] * self.bandit.n
        self._bs = [init_b] * self.bandit.n

    @property
    def estimated_probas(self):
        """
        Estimates the probabilities of each action.

        Returns:
        - estimated_probas (list): Estimated probabilities of each action.
        """
        return [self._as[i] / (self._as[i] + self._bs[i]) for i in range(self.bandit.n)]

    def run_one_step(self):
        """
        Runs one step of the ThompsonSampling solver.

        Returns:
        - i (int): Index of the selected machine to take action on.
        """
        samples = [np.random.beta(self._as[x], self._bs[x]) for x in range(self.bandit.n)]
        i = max(range(self.bandit.n), key=lambda x: samples[x])
        r = self.bandit.generate_reward(i)

        self._as[i] += r
        self._bs[i] += (1 - r)

        return i
