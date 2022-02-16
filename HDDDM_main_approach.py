from math import sqrt
from skmultiflow.drift_detection.base_drift_detector import BaseDriftDetector


class HDDDM(BaseDriftDetector):
    """"
    Implementation of the Hellinger Distance Drift Detection Method as proposed by Ditzler and Polikar.
    Paper: http://users.rowan.edu/~polikar/RESEARCH/PUBLICATIONS/cidue11.pdf
    """

    def __init__(self, distance, features, gamma):
        super().__init__()
        self.prev_dist = 0
        self.lambda_ = 0
        self.batch = 1
        self.sum_eps = 0
        self.sum_eps_sd = 0
        self.dist_diff = 0
        self.epsilon_hat = 0
        self.sigma_hat = 0
        self.beta_hat = 0
        self.gamma = gamma
        self.distance = distance
        self.features = features
        self.reset()

    def reset(self):
        """
        Resets the change detector parameters.
        """
        super().reset()
        self.sum_eps = abs(self.dist_diff)
        self.sum_eps_sd = (abs(self.dist_diff) - self.epsilon_hat) ** 2

    def generate_prop_dic(self, window, union_values):
        """
        Generates dictionaries with proportions per feature. This enables distance computation between windows.
        :param window: The current window of interest.
        :param union_values: list containing union of unique values of the two windows
        :return: Dictionary with proportions per feature.
        """
        dic = {}
        df = window.value_counts()
        n = window.shape[0]
        for key in union_values:
            if key in window.unique():
                dic.update({key: df.loc[key] / n})
            else:
                dic.update({key: 0})
        return dic

    def windows_distance(self, ref_window, current_window):
        """
        Computes the distance between both windows.
        :param ref_window: Reference window
        :param current_window: Current window to compare against the reference window
        :return: total distance between the windows
        """
        actual_dist = 0

        for feature in self.features:
            ref_lst_values = ref_window[feature].unique()
            current_lst_values = current_window[feature].unique()
            union_values = list(set(ref_lst_values) | set(current_lst_values))
            ref_dic = self.generate_prop_dic(ref_window[feature], union_values)
            current_dic = self.generate_prop_dic(current_window[feature], union_values)

            actual_dist += self.distance(ref_dic, current_dic)

        actual_dist /= len(self.features)

        return actual_dist

    def update(self, ref_window, current_window, warn_ratio):
        """
        Updating the drift detector per batch
        :param ref_window: Reference window
        :param current_window: Current window
        :param warn_ratio: The ratio at which possible drift should be warned
        :return: whether or not drift is present in the current batch
        """

        if self.in_concept_change:
            self.reset()

        self.in_concept_change = False
        self.in_warning_zone = False

        actual_dist = self.windows_distance(ref_window, current_window)

        self.dist_diff = actual_dist - self.prev_dist

        self.epsilon_hat = self.sum_eps / (self.batch - self.lambda_)

        self.sigma_hat = sqrt(self.sum_eps_sd / (self.batch - self.lambda_))

        self.beta_hat = self.epsilon_hat + self.gamma * self.sigma_hat

        if abs(self.dist_diff) > self.beta_hat:
            self.lambda_ = self.batch
            self.in_concept_change = True

        elif abs(self.dist_diff) > self.beta_hat * warn_ratio:
            self.sum_eps += abs(self.dist_diff)
            self.sum_eps_sd += (abs(self.dist_diff) - self.epsilon_hat) ** 2
            self.in_warning_zone = True

        else:
            self.sum_eps += abs(self.dist_diff)
            self.sum_eps_sd += (abs(self.dist_diff) - self.epsilon_hat) ** 2

        self.prev_dist = actual_dist
        self.batch += 1

    def add_element(self, input_value):
        """
        Dummy function, as this is needed when using BaseDriftDetector from sk-multiflow.
        However, this function is not usable for this detector, as it only allows for one input variable.
        We use Update instead of Add_element.
        """
        pass