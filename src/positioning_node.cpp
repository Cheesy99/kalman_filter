#include "rclcpp/rclcpp.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"
#include "sensor_msgs/point_cloud2_iterator.hpp"
#include "geometry_msgs/msg/twist.hpp"
#include "tf2/LinearMath/Quaternion.h"
#include "tf2_geometry_msgs/tf2_geometry_msgs.hpp"
#include "kalman_positioning/ukf.hpp"
#include "kalman_positioning/landmark_manager.hpp"


#include <memory>
#include <cmath>

/**
 * @brief Positioning node for UKF-based robot localization (Student Assignment)
 * 
 * This node subscribes to:
 *   - /robot_noisy: Noisy odometry (dead-reckoning)
 *   - /landmarks_observed: Noisy landmark observations
 * 
 * And publishes to:
 *   - /robot_estimated_odometry: Estimated pose and velocity from filter
 * 
 * STUDENT ASSIGNMENT:
 * Implement the Kalman filter logic to fuse odometry and landmark observations
 * to estimate the robot's true position.
 */
class PositioningNode : public rclcpp::Node {
public:
    PositioningNode() : Node("kalman_positioning_node") {
        RCLCPP_INFO(this->get_logger(), "Initializing Kalman Positioning Node");
        
        // Create subscribers
        odometry_sub_ = this->create_subscription<nav_msgs::msg::Odometry>(
            "/robot_noisy",
            rclcpp::QoS(10),
            std::bind(&PositioningNode::odometryCallback, this, std::placeholders::_1)
        );
        RCLCPP_INFO(this->get_logger(), "Subscribed to /robot_noisy");
        
        landmarks_obs_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
            "/landmarks_observed",
            rclcpp::QoS(10),
            std::bind(&PositioningNode::landmarksObservedCallback, this, std::placeholders::_1)
        );
        RCLCPP_INFO(this->get_logger(), "Subscribed to /landmarks_observed");
        
        // Create publisher
        estimated_odom_pub_ = this->create_publisher<nav_msgs::msg::Odometry>(
            "/robot_estimated_odometry", rclcpp::QoS(10)
        );
        RCLCPP_INFO(this->get_logger(), "Publishing to /robot_estimated_odometry");
        
        RCLCPP_INFO(this->get_logger(), "Kalman Positioning Node initialized successfully");

        this->declare_parameter<std::string>("landmarks_csv_path", "");
        this->declare_parameter<double>("process_noise_xy", 1e-4);
        this->declare_parameter<double>("process_noise_theta", 1e-4);
        this->declare_parameter<double>("measurement_noise_xy", 0.01);
        this->declare_parameter<double>("observation_radius", 5.0);

        std::string landmarks_csv =
            this->get_parameter("landmarks_csv_path").as_string();
        double process_noise_xy =
            this->get_parameter("process_noise_xy").as_double();
        double process_noise_theta =
            this->get_parameter("process_noise_theta").as_double();
        double measurement_noise_xy =
            this->get_parameter("measurement_noise_xy").as_double();
        observation_radius_ =
            this->get_parameter("observation_radius").as_double();
            if (!landmark_manager_.loadFromCSV(landmarks_csv)) {
                RCLCPP_ERROR(this->get_logger(),
                             "Failed to load landmarks from CSV: %s",
                             landmarks_csv.c_str());
            } else {
                RCLCPP_INFO(this->get_logger(),
                            "Loaded %zu landmarks from %s",
                            landmark_manager_.getLandmarks().size(),
                            landmarks_csv.c_str());
            }
    
            // Construct UKF with process / measurement noise
            ukf_ = std::make_unique<UKF>(
                process_noise_xy,
                process_noise_theta,
                measurement_noise_xy,
                static_cast<int>(landmark_manager_.getLandmarks().size())
            );
    
            // Give UKF the landmark map
            ukf_->setLandmarks(landmark_manager_.getLandmarks());
    }

private:
    // ============================================================================
    // SUBSCRIBERS AND PUBLISHERS
    // ============================================================================
    
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odometry_sub_;
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr landmarks_obs_sub_;
    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr estimated_odom_pub_;
    
    // ============================================================================
    // PLACEHOLDER: KALMAN FILTER STATE
    // ============================================================================
    // Students should implement a proper Kalman filter (e.g., UKF, EKF) 
    // with the following state:
    //   - Position: x, y (m)
    //   - Orientation: theta (rad)
    //   - Velocity: vx, vy (m/s)
    // And maintain:
    //   - State covariance matrix
    //   - Process noise covariance
    //   - Measurement noise covariance
    
    // ============================================================================
    // CALLBACK FUNCTIONS
    // ============================================================================

    std::unique_ptr<UKF> ukf_;
    LandmarkManager landmark_manager_;

    rclcpp::Time last_odom_time_;
    bool first_odom_ = true;

    // maybe cache params if you want
    double observation_radius_;
    
    /**
     * @brief Callback for noisy odometry measurements
     * 
     * STUDENT TODO:
     * 1. Extract position (x, y) and orientation (theta) from the message
     * 2. Update the Kalman filter's prediction step with this odometry
     * 3. Publish the estimated odometry
     */
    void odometryCallback(const nav_msgs::msg::Odometry::SharedPtr msg) {
        RCLCPP_DEBUG(this->get_logger(), 
            "Odometry received: x=%.3f, y=%.3f", 
            msg->pose.pose.position.x, msg->pose.pose.position.y);
        
        // STUDENT ASSIGNMENT STARTS HERE
        // ========================================================================
        
        if (!ukf_) {
            RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(),
                                 2000, "UKF not initialized yet");
            return;
        }
    
        rclcpp::Time current_time(msg->header.stamp);
    
        // First message: init timestamp only
        if (first_odom_) {
    last_odom_time_ = current_time;
    first_odom_ = false;

    // INITIALIZE STATE FROM ODOMETRY
    x_init_from_odom(msg);

    publishEstimatedOdometry(msg->header.stamp);
    return;
}

    
        double dt = (current_time - last_odom_time_).seconds();
        last_odom_time_ = current_time;
    
        if (dt <= 0.0) {
            return;
        }
    
        double vx = msg->twist.twist.linear.x;
        double vy = msg->twist.twist.linear.y;
        double wz = msg->twist.twist.angular.z;
    
        double dx = vx * dt;
        double dy = vy * dt;
        double dtheta = wz * dt;
    
        ukf_->predict(dt, dx, dy, dtheta);
    
        publishEstimatedOdometry(msg->header.stamp);
    }
    

    void x_init_from_odom(const nav_msgs::msg::Odometry::SharedPtr msg) {
    Eigen::VectorXd x0 = ukf_->getState();

    x0(0) = msg->pose.pose.position.x;
    x0(1) = msg->pose.pose.position.y;
    x0(2) = quaternionToYaw(msg->pose.pose.orientation);
    x0(3) = msg->twist.twist.linear.x;
    x0(4) = msg->twist.twist.linear.y;

    // Directly overwrite state (acceptable for initialization)
    ukf_->setState(x0);
}


    /**
     * @brief Callback for landmark observations
     * 
     * STUDENT TODO:
     * 1. Parse the PointCloud2 data to extract landmark observations
     * 2. Update the Kalman filter's measurement update step with these observations
     * 3. Optionally publish the updated estimated odometry
     */
    void landmarksObservedCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg) {
        if (!ukf_) {
            return;
        }
    
        RCLCPP_DEBUG(this->get_logger(), 
            "Landmark observation received with %u points", msg->width);
    
        // Vector of: (id, meas_x, meas_y, extra)
        std::vector<std::tuple<int, double, double, double>> observations;
    
        try {
            sensor_msgs::PointCloud2ConstIterator<float>    iter_x(*msg, "x");
            sensor_msgs::PointCloud2ConstIterator<float>    iter_y(*msg, "y");
            sensor_msgs::PointCloud2ConstIterator<uint32_t> iter_id(*msg, "id");
            
            for (; iter_x != iter_x.end(); ++iter_x, ++iter_y, ++iter_id) {
                int    landmark_id = static_cast<int>(*iter_id);
                double obs_x       = static_cast<double>(*iter_x);
                double obs_y       = static_cast<double>(*iter_y);
    
                // (Optional) use observation_radius_ to filter
                // double dx = obs_x - ukf_->getState()(0);
                // double dy = obs_y - ukf_->getState()(1);
                // double dist = std::sqrt(dx*dx + dy*dy);
                // if (dist > observation_radius_) continue;
    
                observations.emplace_back(landmark_id, obs_x, obs_y, 0.0);
    
                RCLCPP_DEBUG(this->get_logger(),
                    "Landmark %d observed at (%.3f, %.3f)",
                    landmark_id, obs_x, obs_y);
            }
    
            if (!observations.empty()) {
                ukf_->update(observations);
                // publish updated filter state
                publishEstimatedOdometry(msg->header.stamp);
            }
    
            RCLCPP_DEBUG(this->get_logger(), 
                "Processed %zu landmark observations", observations.size());
    
        } catch (const std::exception& e) {
            RCLCPP_WARN(this->get_logger(), 
                "Failed to parse landmark observations: %s", e.what());
        }
    }
    
    // ============================================================================
    // HELPER FUNCTIONS
    // ============================================================================
    
    /**
     * @brief Convert quaternion to yaw angle
     * @param q Quaternion from orientation
     * @return Yaw angle in radians [-pi, pi]
     */
    double quaternionToYaw(const geometry_msgs::msg::Quaternion& q) {
        tf2::Quaternion tf_q(q.x, q.y, q.z, q.w);
        double roll, pitch, yaw;
        tf2::Matrix3x3(tf_q).getRPY(roll, pitch, yaw);
        return yaw;
    }
    
    /**
     * @brief Normalize angle to [-pi, pi]
     * @param angle Input angle in radians
     * @return Normalized angle in [-pi, pi]
     */
    double normalizeAngle(double angle) {
        while (angle > M_PI) angle -= 2.0 * M_PI;
        while (angle < -M_PI) angle += 2.0 * M_PI;
        return angle;
    }
    
    /**
     * @brief Publish estimated odometry message
     * @param timestamp Message timestamp
     * @param odom_msg Odometry message to publish
     */
    void publishEstimatedOdometry(const rclcpp::Time& timestamp) {
        if (!ukf_) {
            return;
        }
    
        nav_msgs::msg::Odometry estimated_odom;
        estimated_odom.header.stamp = timestamp;
        estimated_odom.header.frame_id = "map";
        estimated_odom.child_frame_id = "robot_estimated";
    
        const Eigen::VectorXd &x = ukf_->getState(); 
    
        estimated_odom.pose.pose.position.x = x(0);
        estimated_odom.pose.pose.position.y = x(1);
        estimated_odom.pose.pose.position.z = 0.0;
    
        tf2::Quaternion q;
        q.setRPY(0.0, 0.0, x(2));
        estimated_odom.pose.pose.orientation = tf2::toMsg(q);
    
        estimated_odom.twist.twist.linear.x = x(3);
        estimated_odom.twist.twist.linear.y = x(4);
        estimated_odom.twist.twist.linear.z = 0.0;
    
    
        estimated_odom_pub_->publish(estimated_odom);
    }
    
};

int main(int argc, char* argv[]) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<PositioningNode>());
    rclcpp::shutdown();
    return 0;
}
