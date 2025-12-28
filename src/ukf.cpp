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
    sigma_points.resize(2 * nx_ + 1);
    sigma_points[0] = mean;
    Eigen::LLT<Eigen::MatrixXd> llt(cov);
    Eigen::MatrixXd L = llt.matrixL();
    Eigen::MatrixXd s = gamma_ * L; 

    for(int i = 0; i < nx_; i++) {
        int k_pos = 1 + i;
        int k_neg = 1 + i + nx_;
      
        sigma_points[k_pos] = mean + s.col(i);
        sigma_points[k_neg] = mean - s.col(i);
       
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
    // =======================================================================
    Eigen::VectorXd new_state = state;
    new_state(0) = new_state(0) + dx;
    new_state(1) = new_state(1) + dy;
    new_state(2) =  this->normalizeAngle(new_state(2) + dtheta);
    if(dt != 0) {
        new_state(3) = dx/dt;
        new_state(4) = dy/dt;
    }
 
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

    std::pair<double, double> landmark = landmarks_.at(landmark_id);
    double delta_x = landmark.first - state(0); 
    double delta_y = landmark.second - state(1);

    Eigen::Vector2d result(delta_x, delta_y);
  
    double theta = normalizeAngle(state(2));

    Eigen::Matrix2d rotationMatrix;
    rotationMatrix(0,0) = std::cos(theta);
    rotationMatrix(0,1) = - std::sin(theta);
    rotationMatrix(1,0) = std::sin(theta);
    rotationMatrix(1,1) = std::cos(theta);

    result = rotationMatrix.transpose() * result;
  

    return result;
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
     
    auto sigma_points = generateSigmaPoints(x_, P_);
    int n_sig = static_cast<int>(sigma_points.size());
    std::vector<Eigen::VectorXd> sigma_points_pred;

    Eigen::VectorXd x_pred = Eigen::VectorXd::Zero(nx_);
    Eigen::MatrixXd P_pred = Eigen::MatrixXd::Zero(nx_, nx_);

   
    double cos_sum = 0.0;
    double sin_sum = 0.0;
    sigma_points_pred.resize(sigma_points.size());

    for (int i = 0; i < n_sig; i++) {
        sigma_points_pred[i] = processModel(sigma_points[i], dt, dx, dy, dtheta);
    
        x_pred += Wm_[i] * sigma_points_pred[i];

        double theta_p = sigma_points_pred[i](2); 
        cos_sum += Wm_[i] * std::cos(theta_p);
        sin_sum += Wm_[i] * std::sin(theta_p);

        Eigen::VectorXd diff = sigma_points_pred[i] - x_pred;
        diff(2) = normalizeAngle(diff(2));
        P_pred += Wc_[i] * diff * diff.transpose();
    }
    
    x_pred(2) = std::atan2(sin_sum, cos_sum);

    P_pred += Q_;
    x_ = x_pred;
    P_ = P_pred;

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
    
     // Process each landmark observation sequentially
     for (const auto& obs : landmark_observations) {
        auto [id, meas_x, meas_y, extra] = obs;

        // 1) Generate sigma points from current state and covariance
        auto sigma_points = generateSigmaPoints(x_, P_);
        int n_sig = static_cast<int>(sigma_points.size());

        // 2) Transform sigma points through measurement model
        std::vector<Eigen::Vector2d> Z_sigma(n_sig);
        Eigen::Vector2d z_pred = Eigen::Vector2d::Zero();

        for (int i = 0; i < n_sig; ++i) {
            Z_sigma[i] = measurementModel(sigma_points[i], id);
            z_pred += Wm_[i] * Z_sigma[i];  // predicted measurement mean
        }

        // 3) Measurement covariance S and cross-covariance Tc
        Eigen::Matrix2d S = Eigen::Matrix2d::Zero();
        Eigen::MatrixXd Tc = Eigen::MatrixXd::Zero(nx_, nz_);  // nx_ x 2

        for (int i = 0; i < n_sig; ++i) {
            Eigen::Vector2d z_diff = Z_sigma[i] - z_pred;

            Eigen::VectorXd x_diff = sigma_points[i] - x_;
            x_diff(2) = normalizeAngle(x_diff(2));  // handle angle in state

            S  += Wc_[i] * z_diff * z_diff.transpose();
            Tc += Wc_[i] * x_diff * z_diff.transpose();
        }

        // Add measurement noise
        S += R_;

        // 4) Kalman gain
        Eigen::MatrixXd K = Tc * S.inverse();  // (nx_ x 2)

        // 5) Innovation (measurement - predicted measurement)
        Eigen::Vector2d z_meas;
        z_meas << meas_x, meas_y;

        Eigen::Vector2d y = z_meas - z_pred;

        // 6) State update
        x_ = x_ + K * y;
        x_(2) = normalizeAngle(x_(2));  // keep theta in a sane range

        // 7) Covariance update
        P_ = P_ - K * S * K.transpose();
    }

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
