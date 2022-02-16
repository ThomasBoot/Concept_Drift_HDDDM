from math import sqrt, log


class Distance:

    def hellinger_dist(self, P, Q):
        """
        P : Dictionary containing proportions for each category in one window
        Q : Dictionary for the next window
        """
        diff = 0
        for key in P.keys():
            diff += (sqrt(P[key]) - sqrt(Q[key])) ** 2
        return 1 / sqrt(2) * sqrt(diff)

    def KL_divergence(self, P, Q):
        """
        This method is used in Jensen_Shannon_divergence
        """
        div = 0
        for key in list(P.keys()):
            if P[key] != 0:  # Otherwise P[key]*logP[key]=0
                div += P[key] * log(P[key] / Q[key])

        return div

    def Jensen_Shannon_divergence(self, P, Q):
        """
        P : Dictionary containing proportions for each category in one window
        Q : Dictionary for the next window
        """
        M = {}
        for key in list(P.keys()):
            M.update({key: (P[key] + Q[key]) / 2})

        return 1 / 2 * (self.KL_divergence(P, M) + self.KL_divergence(Q, M))

    def total_variation_dist(self, P, Q):
        """
        P : Dictionary containing proportions for each category in one window
        Q : Dictionary for the next window
        """

        diff = 0
        for key in P.keys():
            diff += abs(P[key] - Q[key])
        return diff/2