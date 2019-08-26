'''
    File name         : kalman_filter.py
    File Description  : Kalman Filter Algorithm Implementation
    Author            : Srini Ananthakrishnan
    Date created      : 07/14/2017
    Date last modified: 07/16/2017
    Python Version    : 2.7
'''

# Import python libraries
import numpy as np
import cv2
import scipy.linalg

class KalmanFilter(object):
    """
    A simple Kalman filter for tracking bounding boxes in image space.
    The 8-dimensional state space
        x, y, a, h, vx, vy, va, vh
    contains the bounding box center position (x, y), aspect ratio a, height h,
    and their respective velocities.
    Object motion follows a constant velocity model. The bounding box location
    (x, y, a, h) is taken as direct observation of the state space (linear
    observation model).
    """
    def __init__(self):
        ndim, dt = 4, 0.05
        # Create Kalman filter model matrices.
        self._motion_mat = np.eye(2 * ndim, 2 * ndim)
        for i in range(ndim):
            self._motion_mat[i, ndim + i] = dt
        self._update_mat = np.eye(ndim, 2 * ndim)
        # Motion and observation uncertainty are chosen relative to the current
        # state estimate. These weights control the amount of uncertainty in
        # the model. This is a bit hacky.
        self._std_weight_position = 1. / 20
        self._std_weight_velocity = 1. / 160

    def initiate(self, measurement):
        """Create track from unassociated measurement.
        Parameters
        ----------
        measurement : ndarray
            Bounding box coordinates (x, y, a, h) with center position (x, y),
            aspect ratio a, and height h.
        Returns
        -------
        (ndarray, ndarray)
            Returns the mean vector (8 dimensional) and covariance matrix (8x8
            dimensional) of the new track. Unobserved velocities are initialized
            to 0 mean.
        """
        mean_pos = measurement
        mean_vel = np.zeros_like(mean_pos)
        mean = np.r_[mean_pos, mean_vel]
        std = [
            2 * self._std_weight_position * measurement[3],
            2 * self._std_weight_position * measurement[3],
            1e-2,
            2 * self._std_weight_position * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            1e-5,
            10 * self._std_weight_velocity * measurement[3]]
        covariance = np.diag(np.square(std))
        return mean, covariance

    def predict(self, mean, covariance):
        """Run Kalman filter prediction step.
        Parameters
        ----------
        mean : ndarray
            The 8 dimensional mean vector of the object state at the previous
            time step.
        covariance : ndarray
            The 8x8 dimensional covariance matrix of the object state at the
            previous time step.
        Returns
        -------
        (ndarray, ndarray)
            Returns the mean vector and covariance matrix of the predicted
            state. Unobserved velocities are initialized to 0 mean.
        """
        std_pos = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-2,
            self._std_weight_position * mean[3]]
        std_vel = [
            self._std_weight_velocity * mean[3],
            self._std_weight_velocity * mean[3],
            1e-5,
            self._std_weight_velocity * mean[3]]
        motion_cov = np.diag(np.square(np.r_[std_pos, std_vel]))
        mean = np.dot(self._motion_mat, mean)
        covariance = np.linalg.multi_dot((
            self._motion_mat, covariance, self._motion_mat.T)) + motion_cov
        return mean, covariance

    def project(self, mean, covariance):
        """Project state distribution to measurement space.
        Parameters
        ----------
        mean : ndarray
            The state's mean vector (8 dimensional array).
        covariance : ndarray
            The state's covariance matrix (8x8 dimensional).
        Returns
        -------
        (ndarray, ndarray)
            Returns the projected mean and covariance matrix of the given state
            estimate.
        """
        std = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-1,
            self._std_weight_position * mean[3]]
        innovation_cov = np.diag(np.square(std))
        mean = np.dot(self._update_mat, mean)
        covariance = np.linalg.multi_dot((
            self._update_mat, covariance, self._update_mat.T))
        return mean, covariance + innovation_cov

    def update(self, mean, covariance, measurement):
        """Run Kalman filter correction step.
        Parameters
        ----------
        mean : ndarray
            The predicted state's mean vector (8 dimensional).
        covariance : ndarray
            The state's covariance matrix (8x8 dimensional).
        measurement : ndarray
            The 4 dimensional measurement vector (x, y, a, h), where (x, y)
            is the center position, a the aspect ratio, and h the height of the
            bounding box.
        Returns
        -------
        (ndarray, ndarray)
            Returns the measurement-corrected state distribution.
        """
        projected_mean, projected_cov = self.project(mean, covariance)
        chol_factor, lower = scipy.linalg.cho_factor(
            projected_cov, lower=True, check_finite=False)
        kalman_gain = scipy.linalg.cho_solve(
            (chol_factor, lower), np.dot(covariance, self._update_mat.T).T,
            check_finite=False).T
        innovation = measurement - projected_mean
        new_mean = mean + np.dot(innovation, kalman_gain.T)
        new_covariance = covariance - np.linalg.multi_dot((
            kalman_gain, projected_cov, kalman_gain.T))
        return new_mean, new_covariance

class KalmanFilter_Center(object):
    """Kalman Filter class keeps track of the estimated state of
    the system and the variance or uncertainty of the estimate.
    Predict and Correct methods implement the functionality
    Reference: https://en.wikipedia.org/wiki/Kalman_filter
    Attributes: None
    """

    def __init__(self):
        """Initialize variable used by Kalman Filter class
        Args:
            None
        Return:
            None
        """
        self.dt = 0.005  # delta time

        self.A = np.array([[1, 0], [0, 1]])  # matrix in observation equations
        self.u = np.zeros((2, 1))  # previous state vector

        # (x,y) tracking object center
        self.b = np.array([[0], [255]])  # vector of observations

        self.P = np.diag((3.0, 3.0))  # covariance matrix
        self.F = np.array([[1.0, self.dt], [0.0, 1.0]])  # state transition mat

        self.Q = np.eye(self.u.shape[0])  # process noise matrix
        self.R = np.eye(self.b.shape[0])  # observation noise matrix
        self.lastResult = np.array([[0], [255]])

    def predict(self):
        """Predict state vector u and variance of uncertainty P (covariance).
            where,
            u: previous state vector
            P: previous covariance matrix
            F: state transition matrix
            Q: process noise matrix
        Equations:
            u'_{k|k-1} = Fu'_{k-1|k-1}
            P_{k|k-1} = FP_{k-1|k-1} F.T + Q
            where,
                F.T is F transpose
        Args:
            None
        Return:
            vector of predicted state estimate
        """
        # Predicted state estimate
        self.u = np.round(np.dot(self.F, self.u))
        # Predicted estimate covariance
        self.P = np.dot(self.F, np.dot(self.P, self.F.T)) + self.Q
        self.lastResult = self.u  # same last predicted result
        return self.u

    def correct(self, b, flag):
        """Correct or update state vector u and variance of uncertainty P (covariance).
        where,
        u: predicted state vector u
        A: matrix in observation equations
        b: vector of observations
        P: predicted covariance matrix
        Q: process noise matrix
        R: observation noise matrix
        Equations:
            C = AP_{k|k-1} A.T + R
            K_{k} = P_{k|k-1} A.T(C.Inv)
            u'_{k|k} = u'_{k|k-1} + K_{k}(b_{k} - Au'_{k|k-1})
            P_{k|k} = P_{k|k-1} - K_{k}(CK.T)
            where,
                A.T is A transpose
                C.Inv is C inverse
        Args:
            b: vector of observations
            flag: if "true" prediction result will be updated else detection
        Return:
            predicted state vector u
        """

        if not flag:  # update using prediction
            self.b = self.lastResult
        else:  # update using detection
            self.b = b
        C = np.dot(self.A, np.dot(self.P, self.A.T)) + self.R
        K = np.dot(self.P, np.dot(self.A.T, np.linalg.inv(C)))

        self.u = np.round(self.u + np.dot(K, (self.b - np.dot(self.A,
                                                              self.u))))
        self.P = self.P - np.dot(K, np.dot(C, K.T))
        self.lastResult = self.u
        return self.u

class KalmanFilter_cv(object):
    def __init__(self):
        self.kalman = cv2.KalmanFilter(4, 2, 0)
        self.state = np.array([0, 0, 0, 0], dtype='float64')  # initial position
        self.kalman.transitionMatrix = np.array([[1., 0., .1, 0.],
                                            [0., 1., 0., .1],
                                            [0., 0., 1., 0.],
                                            [0., 0., 0., 1.]])
        self.kalman.measurementMatrix = 1. * np.eye(2, 4)
        self.kalman.processNoiseCov = 1e-5 * np.eye(4, 4)
        self.kalman.measurementNoiseCov = 1e-3 * np.eye(2, 2)
        self.kalman.errorCovPost = 1e-1 * np.eye(4, 4)
        self.kalman.statePost = self.state
    def predict(self):
        self.prediction = self.kalman.predict()
    def correct(self,p_scalar):
        #print("p shape ",np.shape(p_scalar))
        c,r,w,h = p_scalar
        if w != 0 and h != 0:
            self.state = np.array([c + w / 2, r + h / 2, 0, 0], dtype='float64')
            # kalman.statePost = state
            measurement = (np.dot(self.kalman.measurementNoiseCov, np.random.randn(2, 1))).reshape(-1)
            measurement = np.dot(self.kalman.measurementMatrix, self.state) + measurement
            posterior = self.kalman.correct(measurement)
            pos = (posterior[0], posterior[1])
        else:
            measurement = (np.dot(self.kalman.measurementNoiseCov, np.random.randn(2, 1))).reshape(-1)
            measurement = np.dot(self.kalman.measurementMatrix, self.state) + measurement
            pos = (self.prediction[0], self.prediction[1])
        process_noise = np.sqrt(self.kalman.processNoiseCov[0, 0]) * np.random.randn(4, 1)
        self.state = np.dot(self.kalman.transitionMatrix, self.state) + process_noise.reshape(-1)
        return pos