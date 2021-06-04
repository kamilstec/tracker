import numpy as np
import scipy.linalg

class KalmanFilter(object):
    def __init__(self):
        ndim, dt = 4, 1.

        self.motion_model = np.eye(2 * ndim, 2 * ndim)
        for i in range(ndim):
            self.motion_model[i, ndim + i] = dt
        self.observation_model = np.eye(ndim, 2 * ndim)

        self.std_weight_position = 1. / 20
        self.std_weight_velocity = 1. / 160

    def initiate(self, measurement):
        mean_pos = measurement
        mean_vel = np.zeros_like(mean_pos)
        mean = np.r_[mean_pos, mean_vel]

        std_pos = [
            2 * self.std_weight_position * measurement[3],
            2 * self.std_weight_position * measurement[3],
            1e-2,
            2 * self.std_weight_position * measurement[3]]
        std_vel = [
            10 * self.std_weight_velocity * measurement[3],
            10 * self.std_weight_velocity * measurement[3],
            1e-5,
            10 * self.std_weight_velocity * measurement[3]]
        covariance = np.diag(np.square(np.r_[std_pos, std_vel]))

        return mean, covariance

    def predict(self, mean, covariance):
        std_pos = [
            self.std_weight_position * mean[3],
            self.std_weight_position * mean[3],
            1e-2,
            self.std_weight_position * mean[3]]
        std_vel = [
            self.std_weight_velocity * mean[3],
            self.std_weight_velocity * mean[3],
            1e-5,
            self.std_weight_velocity * mean[3]]
        motion_cov = np.diag(np.square(np.r_[std_pos, std_vel])) # Q_t

        # 1) - pierwszy krok w algorytmie filtru Kalmana
        mean = np.dot(self.motion_model, mean)
        # 2)
        covariance = np.linalg.multi_dot((
            self.motion_model, covariance, self.motion_model.T)) + motion_cov

        return mean, covariance

    def update(self, mean, covariance, measurement):
        std_pos = [
            self.std_weight_position * mean[3],
            self.std_weight_position * mean[3],
            1e-1,
            self.std_weight_position * mean[3]]
        observation_cov = np.diag(np.square(std_pos)) # R_t
        # 3)
        projected_cov = np.linalg.multi_dot((
            self.observation_model, covariance, self.observation_model.T)) + observation_cov
        chol_factor, lower = scipy.linalg.cho_factor(
            projected_cov, lower=True, check_finite=False)
        kalman_gain = scipy.linalg.cho_solve(
            (chol_factor, lower), np.dot(covariance, self.observation_model.T).T,
            check_finite=False).T
        # 4)
        projected_mean = np.dot(self.observation_model, mean)
        innovation = measurement - projected_mean
        new_mean = mean + np.dot(innovation, kalman_gain.T)
        # 5)
        new_covariance = covariance - np.linalg.multi_dot((
            kalman_gain, projected_cov, kalman_gain.T))

        return new_mean, new_covariance
