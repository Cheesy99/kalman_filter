#include "kalman_positioning/ukf.hpp"
#include <iostream>
#include <map>

/**
 * @brief Initializes the Unscented Kalman Filter.
 *
 * Sets up the state covariance (P), process noise (Q), measurement noise (R),
 * and calculates the sigma point weights (Wm, Wc) based on UKF hyperparameters.
 *
 * @param process_noise_xy Standard deviation of position noise (meters).
 * @param process_noise_theta Standard deviation of heading noise (radians).
 * @param measurement_noise_xy Standard deviation of measurement noise (meters).
 * @param num_landmarks Expected number of landmarks (currently reserved).
 */
UKF::UKF(double process_noise_xy, double process_noise_theta,
         double measurement_noise_xy, int num_landmarks)
    : nx_(5), nz_(2) {

  this->lambda_ = ALPHA * ALPHA * (nx_ + KAPPA) - nx_;
  this->gamma_ = std::sqrt(nx_ + lambda_);
  this->x_ = Eigen::VectorXd::Zero(nx_);
  this->P_ = Eigen::MatrixXd::Identity(nx_, nx_) * 1.0;

  this->Q_ = Eigen::MatrixXd::Zero(nx_, nx_);
  this->Q_(0, 0) = process_noise_xy;
  this->Q_(1, 1) = process_noise_xy;
  this->Q_(2, 2) = process_noise_theta;

  this->R_ = measurement_noise_xy * Eigen::MatrixXd::Identity(nz_, nz_);

  int numberOfSigmaPoints = 2 * nx_ + 1;
  this->Wm_.resize(numberOfSigmaPoints);
  this->Wc_.resize(numberOfSigmaPoints);

  double c = nx_ + lambda_;
  Wm_[0] = lambda_ / c;
  Wc_[0] = lambda_ / c + (1.0 - ALPHA * ALPHA + BETA);

  double wi = 1.0 / (2.0 * c);
  for (int i = 1; i < numberOfSigmaPoints; ++i) {
    Wm_[i] = wi;
    Wc_[i] = wi;
  }
}

/**
 * @brief Generates sigma points around the current mean using the covariance
 * matrix.
 *
 * This method performs a Cholesky decomposition (LLT). If the covariance matrix
 * is not positive definite due to numerical error, a small "jitter" is added to
 * the diagonal.
 *
 * @param mean Current state mean.
 * @param cov Current state covariance.
 * @return std::vector<Eigen::VectorXd> Vector of 2*nx+1 sigma points.
 */
std::vector<Eigen::VectorXd>
UKF::generateSigmaPoints(const Eigen::VectorXd &mean,
                         const Eigen::MatrixXd &cov) {
  std::vector<Eigen::VectorXd> sigma_points;
  sigma_points.resize(2 * nx_ + 1);
  sigma_points[0] = mean;

  Eigen::MatrixXd cov_sym = 0.5 * (cov + cov.transpose());
  Eigen::LLT<Eigen::MatrixXd> llt(cov_sym);

  // Fallback: Add jitter if decomposition fails (matrix not positive definite)
  if (llt.info() != Eigen::Success) {
    Eigen::MatrixXd cov_jitter = cov_sym;
    cov_jitter.diagonal().array() += 1e-6;
    llt.compute(cov_jitter);
  }

  Eigen::MatrixXd L = llt.matrixL();
  Eigen::MatrixXd s = gamma_ * L;

  for (int i = 0; i < nx_; i++) {
    sigma_points[1 + i] = mean + s.col(i);
    sigma_points[1 + i + nx_] = mean - s.col(i);
  }
  return sigma_points;
}

/**
 * @brief Process Model (Motion Model).
 *
 * Propagates a single state vector forward in time based on odometry inputs.
 *
 * @param state Initial state.
 * @param dt Time delta.
 * @param dx, dy, dtheta Odometry deltas.
 * @return Eigen::VectorXd Predicted state.
 */
Eigen::VectorXd UKF::processModel(const Eigen::VectorXd &state, double dt,
                                  double dx, double dy, double dtheta) {
  Eigen::VectorXd new_state = state;
  new_state(0) += dx;
  new_state(1) += dy;
  new_state(2) = this->normalizeAngle(new_state(2) + dtheta);

  if (dt > 1e-6) {
    new_state(3) = dx / dt;
    new_state(4) = dy / dt;
  }
  return new_state;
}

/**
 * @brief Measurement Model.
 *
 * Transforms a state vector into expected landmark observations (relative x, y)
 * in the robot's local frame.
 *
 * @param state The state vector to transform.
 * @param landmark_id The ID of the landmark being observed.
 * @return Eigen::Vector2d Expected measurement [x_rel, y_rel].
 */
Eigen::Vector2d UKF::measurementModel(const Eigen::VectorXd &state,
                                      int landmark_id) {
  auto it = landmarks_.find(landmark_id);
  if (it == landmarks_.end())
    return Eigen::Vector2d::Zero();

  const double dx = it->second.first - state(0);
  const double dy = it->second.second - state(1);
  const double th = state(2);

  const double c = std::cos(th);
  const double s = std::sin(th);

  Eigen::Vector2d z;
  z(0) = c * dx + s * dy;
  z(1) = -s * dx + c * dy;
  return z;
}

/**
 * @brief Normalizes an angle to be within the interval [-PI, PI].
 */
double UKF::normalizeAngle(double angle) {
  while (angle > M_PI)
    angle -= 2.0 * M_PI;
  while (angle < -M_PI)
    angle += 2.0 * M_PI;
  return angle;
}

/**
 * @brief Predicts the next state using the Unscented Transform.
 *
 * Generates sigma points, propagates them through the process model,
 * and recovers the predicted mean and covariance.
 */
void UKF::predict(double dt, double dx, double dy, double dtheta) {
  auto sigma_points = generateSigmaPoints(x_, P_);
  int n_sig = static_cast<int>(sigma_points.size());

  std::vector<Eigen::VectorXd> sigma_points_pred(n_sig);
  Eigen::VectorXd x_pred = Eigen::VectorXd::Zero(nx_);
  Eigen::MatrixXd P_pred = Eigen::MatrixXd::Zero(nx_, nx_);

  double cos_sum = 0.0, sin_sum = 0.0;

  for (int i = 0; i < n_sig; i++) {
    sigma_points_pred[i] = processModel(sigma_points[i], dt, dx, dy, dtheta);
    x_pred(0) += Wm_[i] * sigma_points_pred[i](0);
    x_pred(1) += Wm_[i] * sigma_points_pred[i](1);
    x_pred(3) += Wm_[i] * sigma_points_pred[i](3);
    x_pred(4) += Wm_[i] * sigma_points_pred[i](4);

    // Compute angular mean using vector sums to avoid wrap-around issues
    cos_sum += Wm_[i] * std::cos(sigma_points_pred[i](2));
    sin_sum += Wm_[i] * std::sin(sigma_points_pred[i](2));
  }

  x_pred(2) = std::atan2(sin_sum, cos_sum);

  for (int i = 0; i < n_sig; i++) {
    Eigen::VectorXd diff = sigma_points_pred[i] - x_pred;
    diff(2) = normalizeAngle(diff(2));
    P_pred += Wc_[i] * diff * diff.transpose();
  }
  P_ = P_pred + Q_;
  x_ = x_pred;
}

/**
 * @brief Updates the state based on landmark observations.
 *
 * Performs an outlier rejection test (Mahalanobis distance > 9.21) before
 * update.
 *
 * @param landmark_observations Tuple of <id, meas_x, meas_y, extra>.
 */
void UKF::update(const std::vector<std::tuple<int, double, double, double>>
                     &landmark_observations) {
  for (const auto &obs : landmark_observations) {
    auto [id, meas_x, meas_y, extra] = obs;
    if (!hasLandmark(id))
      continue;

    auto sigma_points = generateSigmaPoints(x_, P_);
    int n_sig = static_cast<int>(sigma_points.size());

    std::vector<Eigen::Vector2d> Z_sigma(n_sig);
    Eigen::Vector2d z_pred = Eigen::Vector2d::Zero();

    for (int i = 0; i < n_sig; ++i) {
      Z_sigma[i] = measurementModel(sigma_points[i], id);
      z_pred += Wm_[i] * Z_sigma[i];
    }

    Eigen::Matrix2d S = Eigen::Matrix2d::Zero();
    Eigen::MatrixXd Tc = Eigen::MatrixXd::Zero(nx_, nz_);

    for (int i = 0; i < n_sig; ++i) {
      Eigen::Vector2d z_diff = Z_sigma[i] - z_pred;
      Eigen::VectorXd x_diff = sigma_points[i] - x_;
      x_diff(2) = normalizeAngle(x_diff(2));

      S += Wc_[i] * z_diff * z_diff.transpose();
      Tc += Wc_[i] * x_diff * z_diff.transpose();
    }

    S += R_;
    Eigen::LDLT<Eigen::Matrix2d> ldlt(S);
    if (ldlt.info() != Eigen::Success)
      continue;

    Eigen::Vector2d z_meas(meas_x, meas_y);
    Eigen::Vector2d y = z_meas - z_pred;

    // Chi-square test (99% confidence, 2 DOF) to reject outliers
    if (y.transpose() * ldlt.solve(y) > 9.21)
      continue;

    Eigen::MatrixXd K = Tc * ldlt.solve(Eigen::Matrix2d::Identity());
    x_ += K * y;
    x_(2) = normalizeAngle(x_(2));
    P_ -= K * S * K.transpose();
  }
}

/**
 * @brief Sets the global map of landmarks.
 */
void UKF::setLandmarks(
    const std::map<int, std::pair<double, double>> &landmarks) {
  landmarks_ = landmarks;
}

/**
 * @brief Checks if a landmark ID exists in the map.
 */
bool UKF::hasLandmark(int id) const {
  return landmarks_.find(id) != landmarks_.end();
}