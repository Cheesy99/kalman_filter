#ifndef KALMAN_POSITIONING_UKF_HPP
#define KALMAN_POSITIONING_UKF_HPP

#include <Eigen/Dense>
#include <vector>
#include <map>
#include <cmath>

class UKF {
public:
    /**
     * @brief Unscented Kalman Filter for robot localization with landmarks
     * State: [x, y, theta, vx, vy]
     */
    
    UKF(double process_noise_xy, double process_noise_theta,
        double measurement_noise_xy, int num_landmarks);
    
    /**
     * @brief Predict step using odometry
     * @param dt Time step
     * @param dx Linear displacement in x
     * @param dy Linear displacement in y
     * @param dtheta Angular displacement
     */
    void predict(double dt, double dx, double dy, double dtheta);
    
    void setState(const Eigen::VectorXd& x) { x_ = x; }


    /**
     * @brief Update step with landmark observations
     * @param landmark_observations Vector of {id, x, y, noise_cov}
     */
    void update(const std::vector<std::tuple<int, double, double, double>>& landmark_observations);
    
    /**
     * @brief Set landmark positions
     * @param landmarks Map of landmark_id -> {x, y}
     */
    void setLandmarks(const std::map<int, std::pair<double, double>>& landmarks);
    
    /**
     * @brief Get estimated state
     * @return State vector [x, y, theta, vx, vy]
     */
    Eigen::VectorXd getState() const { return x_; }
    
    /**
     * @brief Get state covariance
     */
    Eigen::MatrixXd getCovariance() const { return P_; }
    
    /**
     * @brief Get position estimate
     */
    std::pair<double, double> getPosition() const { return {x_(0), x_(1)}; }
    
    /**
     * @brief Get orientation estimate
     */
    double getOrientation() const { return x_(2); }
    
    /**
     * @brief Check if landmark exists
     */
    bool hasLandmark(int id) const;

private:
    static constexpr double ALPHA = 1e-3;  // Spread of sigma points
    static constexpr double BETA = 2.0;     // Distribution info (Gaussian)
    static constexpr double KAPPA = 0.0;    // Secondary scaling
    
    int nx_;  // State dimension
    int nz_;  // Measurement dimension per landmark
    double lambda_;
    double gamma_;
    
    Eigen::VectorXd x_;  // State estimate
    Eigen::MatrixXd P_;  // State covariance
    Eigen::MatrixXd Q_;  // Process noise covariance
    Eigen::MatrixXd R_;  // Measurement noise covariance
    
    std::map<int, std::pair<double, double>> landmarks_;  // Known landmark positions
    std::vector<double> Wm_;  // Weights for mean
    std::vector<double> Wc_;  // Weights for covariance
    
    /**
     * @brief Generate sigma points
     */
    std::vector<Eigen::VectorXd> generateSigmaPoints(const Eigen::VectorXd& mean,
                                                      const Eigen::MatrixXd& cov);
    
    /**
     * @brief Process model (state transition)
     */
    Eigen::VectorXd processModel(const Eigen::VectorXd& state, double dt,
                                 double dx, double dy, double dtheta);
    
    /**
     * @brief Measurement model for landmark observation
     */
    Eigen::Vector2d measurementModel(const Eigen::VectorXd& state, int landmark_id);
    
    /**
     * @brief Normalize angle to [-pi, pi]
     */
    static double normalizeAngle(double angle);
};

#endif  // KALMAN_POSITIONING_UKF_HPP
