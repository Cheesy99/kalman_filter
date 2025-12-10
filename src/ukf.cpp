#include "kalman_positioning/ukf.hpp"
#include <iostream>
#include <map>

/**
 * STUDENT ASSIGNMENT: Unscented Kalman Filter Implementation
 * 
 * This file contains placeholder implementations for the UKF class methods.
 * Students should implement each method according to the UKF algorithm.
 * 
 * Reference: Wan, E. A., & Van Der Merwe, R. (2000). 
 * "The Unscented Kalman Filter for Nonlinear Estimation"
 */

// ============================================================================
// CONSTRUCTOR
// ============================================================================

/**
 * @brief Initialize the Unscented Kalman Filter
 * 
 * STUDENT TODO:
 * 1. Initialize filter parameters (alpha, beta, kappa, lambda)
 * 2. Initialize state vector x_ with zeros
 * 3. Initialize state covariance matrix P_ 
 * 4. Set process noise covariance Q_
 * 5. Set measurement noise covariance R_
 * 6. Calculate sigma point weights for mean and covariance
 */
UKF::UKF(double process_noise_xy, double process_noise_theta,
         double measurement_noise_xy, int num_landmarks)
    : nx_(5), nz_(2) {
    
    // STUDENT IMPLEMENTATION STARTS HERE
    // ========================================================================
        this->lambda_ = ALPHA * ALPHA * (nx_ + KAPPA) - nx_;
        this->gamma_ = std::sqrt(nx_ + lambda_);
        this->x_ = Eigen::VectorXd::Zero(nx_);
        this->P_ = Eigen::MatrixXd::Identity(nx_, nx_);

        this->Q_ = Eigen::MatrixXd::Zero(nx_, nx_);
        this->Q_(0,0) = process_noise_xy;
        this->Q_(1,1) = process_noise_xy;
        this->Q_(2,2) = process_noise_theta;

        this->R_ = measurement_noise_xy * Eigen::MatrixXd::Identity(nz_, nz_);

        // ============================================================================
        // TODO: Really understand the math below why we use these values hwo the work
        // ============================================================================
        int numberOfSigmaPoints = 2 * nx_ + 1;
        this->Wm_.resize(numberOfSigmaPoints);
        this->Wc_.resize(numberOfSigmaPoints);
       
        double c   = nx_ + lambda_;

        Wm_[0] = lambda_ / c;
        Wc_[0] = lambda_ / c + (1.0 - ALPHA * ALPHA + BETA);

        double wi = 1.0 / (2.0 * c);

        for(int i = 1; i < numberOfSigmaPoints; ++i){
            Wm_[i] = wi;
            Wc_[i] = wi;
        }
}

// ============================================================================
// SIGMA POINT GENERATION
// ============================================================================

/**
 * @brief Generate sigma points from mean and covariance
 * 
 * STUDENT TODO:
 * 1. Start with the mean as the first sigma point
 * 2. Compute Cholesky decomposition of covariance
 * 3. Generate 2*n symmetric sigma points around the mean
 */
std::vector<Eigen::VectorXd> UKF::generateSigmaPoints(const Eigen::VectorXd& mean,
                                                       const Eigen::MatrixXd& cov) {
    // STUDENT IMPLEMENTATION STARTS HERE
    // ========================================================================
    

    std::vector<Eigen::VectorXd> sigma_points;
    sigma_points.resize(2 * mean.size() + 1);
    sigma_points[0] = mean;
                                         
    for(int i = 0; i < mean.size() - 1; i++) {
        int k_pos = 1 + i;
        int k_neg = 1 + i + mean.size();
    
            sigma_points[k_pos] = mean + cov.col(i - 1);
            sigma_points[k_neg] = mean - cov.col(i - 1);
       
    }
    
    return sigma_points;
}

// ============================================================================
// PROCESS MODEL
// ============================================================================

/**
 * @brief Apply motion model to a state vector
 * 
 * STUDENT TODO:
 * 1. Updates position: x' = x + dx, y' = y + dy
 * 2. Updates orientation: theta' = theta + dtheta (normalized)
 * 3. Updates velocities: vx' = dx/dt, vy' = dy/dt
 */
Eigen::VectorXd UKF::processModel(const Eigen::VectorXd& state, double dt,
                                  double dx, double dy, double dtheta) {
    // STUDENT IMPLEMENTATION STARTS HERE
    // ========================================================================
    
    Eigen::VectorXd new_state = state;
    
    return new_state;
}

// ============================================================================
// MEASUREMENT MODEL
// ============================================================================

/**
 * @brief Predict measurement given current state and landmark
 * 
 * STUDENT TODO:
 * 1. Calculate relative position: landmark - robot position
 * 2. Transform to robot frame using robot orientation
 * 3. Return relative position in robot frame
 */
Eigen::Vector2d UKF::measurementModel(const Eigen::VectorXd& state, int landmark_id) {
    // STUDENT IMPLEMENTATION STARTS HERE
    // ========================================================================
    
    if (landmarks_.find(landmark_id) == landmarks_.end()) {
        return Eigen::Vector2d::Zero();
    }
    
    return Eigen::Vector2d::Zero();
}

// ============================================================================
// ANGLE NORMALIZATION
// ============================================================================

double UKF::normalizeAngle(double angle) {
    while (angle > M_PI) angle -= 2.0 * M_PI;
    while (angle < -M_PI) angle += 2.0 * M_PI;
    return angle;
}

// ============================================================================
// PREDICTION STEP
// ============================================================================

/**
 * @brief Kalman Filter Prediction Step (Time Update)
 * 
 * STUDENT TODO:
 * 1. Generate sigma points from current state and covariance
 * 2. Propagate each sigma point through motion model
 * 3. Calculate mean and covariance of predicted sigma points
 * 4. Add process noise
 * 5. Update state and covariance estimates
 */
void UKF::predict(double dt, double dx, double dy, double dtheta) {
    // STUDENT IMPLEMENTATION STARTS HERE
    // ========================================================================
    
    std::cout << "UKF Predict: TODO - Implement prediction step" << std::endl;
}

// ============================================================================
// UPDATE STEP
// ============================================================================

/**
 * @brief Kalman Filter Update Step (Measurement Update)
 * 
 * STUDENT TODO:
 * 1. Generate sigma points
 * 2. Transform through measurement model
 * 3. Calculate predicted measurement mean
 * 4. Calculate measurement and cross-covariance
 * 5. Compute Kalman gain
 * 6. Update state with innovation
 * 7. Update covariance
 */
void UKF::update(const std::vector<std::tuple<int, double, double, double>>& landmark_observations) {
    if (landmark_observations.empty()) {
        return;
    }
    
    // STUDENT IMPLEMENTATION STARTS HERE
    // ========================================================================
    
    std::cout << "UKF Update: TODO - Implement measurement update step" << std::endl;
}

// ============================================================================
// LANDMARK MANAGEMENT
// ============================================================================

void UKF::setLandmarks(const std::map<int, std::pair<double, double>>& landmarks) {
    landmarks_ = landmarks;
}

bool UKF::hasLandmark(int id) const {
    return landmarks_.find(id) != landmarks_.end();
}
