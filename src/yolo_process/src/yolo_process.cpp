#include <detect_msg/msg/nn_detect_data.hpp>
#include <detect_msg/msg/nn_detect_array.hpp>
#include <opencv2/opencv.hpp>
#include <rclcpp/rclcpp.hpp>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/msg/image.hpp>
#include <vector>
#include <Eigen/Dense>
#include <cmath>

using NNDetectData = detect_msg::msg::NNDetectData;
using NNDetectArray = detect_msg::msg::NNDetectArray;

// 适用于二维的作匀速直线运动的物体
class KalmanFilter2D
{
    /*
        假设2维像素坐标系下，装甲板中心作匀速直线运动
        状态向量Xk = [Xk,Vx-k,Yk,Vy-k]T
        Xk = Xk-1 + Vx,k-1*delta(t)
        Vx,k = Vx,k-1
        已知可测量量为装甲板中心的x和y坐标
    */
private:
    // 状态更新矩阵A（4*4）
    Eigen::Matrix4f A;

    // 过程噪声协方差矩阵Q（4*4）
    Eigen::Matrix4f Q;

    // 卡尔曼增益K(4*2)（状态维度×观测维度）
    Eigen::Matrix<float, 4, 2> Kk;

    // 观测矩阵H（2*4）
    Eigen::Matrix<float, 2, 4, Eigen::RowMajor> H;

    // 装甲板中心点
    cv::Point2f armor_center;

    // 观测向量Zk(2*1)
    Eigen::Vector2f Zk;

    // 先验预测状态矩阵X^-k(4*1),4维列向量（x, vx, y, vy）
    Eigen::Vector4f prior_Xk_predict;

    // 后验预测状态矩阵X^k(4*1),4维列向量(x,vx,y,vy)
    Eigen::Vector4f posterior_Xk_predict;

    // 先验预测协方差矩阵P^-k(4*4)
    Eigen::Matrix4f prior_Pk_predict;

    // 后验预测协方差矩阵P^k(4*4)
    Eigen::Matrix4f posterior_Pk_predict;

    // 测量噪声协方差矩阵R(2*2)
    Eigen::Matrix2f R;

    // 时间间隔
    double kalman_dt = 0.0;

    // 加速度噪声方差
    float a_variance = 0.1;

public:
    KalmanFilter2D(cv::Point2f init_center, double dt)
    {

        // 初始化时间间隔
        kalman_dt = dt;

        // 初始化状态矩阵
        posterior_Xk_predict << init_center.x,
            0.0f,
            init_center.y,
            0.0f;
        // 初始化状态协方差矩阵
        posterior_Pk_predict = Eigen::Matrix4f::Identity() * 1e-2;

        // 初始化状态更新矩阵A（匀速直线运动）
        A << 1, dt, 0, 0,
            0, 1, 0, 0,
            0, 0, 1, dt,
            0, 0, 0, 1;

        // 初始化过程噪声矩阵Q
        float dt2 = dt * dt;
        float dt3 = dt2 * dt;
        float dt4 = dt3 * dt;
        Q << dt4 / 4, dt3 / 2, 0, 0,
            dt3 / 2, dt2, 0, 0,
            0, 0, dt4 / 4, dt3 / 2,
            0, 0, dt3 / 2, dt2;
        Q *= a_variance;

        // 初始化观测矩阵H
        H << 1, 0, 0, 0,
            0, 0, 1, 0;

        // 初始化测量噪声矩阵R
        R << 0.1f, 0,
            0, 0.1f;
    }

    // -------------------------- 新增：仅执行预测步骤（无观测值时调用） --------------------------
    // 作用：根据上一帧状态预测当前帧状态（无观测值时，保持速度和协方差演进）
    void predict(double new_dt)
    {
        // 更新时间间隔（适配帧率变化）
        kalman_dt = new_dt;

        // 1. 更新状态转移矩阵A（dt可能变化，需重新计算）
        A(0, 1) = kalman_dt; // x = x + vx*dt
        A(2, 3) = kalman_dt; // y = y + vy*dt

        // 2. 预测先验状态（x_prior = A * x_posterior）
        prior_Xk_predict = A * posterior_Xk_predict;

        // 3. 预测先验协方差（P_prior = A * P_posterior * A^T + Q）
        prior_Pk_predict = A * posterior_Pk_predict * A.transpose() + Q;
    }

    // -------------------------- 新增：执行更新步骤（有观测值时调用） --------------------------
    // 作用：用当前帧观测值（装甲板中心点）修正预测状态
    void update(cv::Point2f observed_center)
    {
        // 1. 构造观测向量Zk（2x1：x, y）
        Zk << observed_center.x, observed_center.y;

        // 2. 计算卡尔曼增益Kk（4x2）
        Eigen::Matrix2f S = H * prior_Pk_predict * H.transpose() + R; // 观测残差协方差（2x2）
        Kk = prior_Pk_predict * H.transpose() * S.inverse();          // K = P_prior * H^T * S^{-1}

        // 3. 修正后验状态（x_posterior = x_prior + K*(Z - H*x_prior)）
        posterior_Xk_predict = prior_Xk_predict + Kk * (Zk - H * prior_Xk_predict);

        // 4. 修正后验协方差（P_posterior = (I - K*H) * P_prior）
        posterior_Pk_predict = (Eigen::Matrix4f::Identity() - Kk * H) * prior_Pk_predict;
    }

    // 懂得都懂
    /*
    void predict_process(cv::Point2f found_center)
    {
        // 预测状态
        prior_Xk_predict = A * posterior_Xk_predict;

        // 测量值
        Zk << found_center.x, found_center.y;

        // 预测协方差
        prior_Pk_predict = A * posterior_Pk_predict * A.transpose() + Q;

        // 计算卡尔曼增益
        Kk = (prior_Pk_predict * H.transpose()) * (H * prior_Pk_predict * H.transpose() + R).inverse();

        // 得到后验状态
        posterior_Xk_predict = prior_Xk_predict + Kk * (Zk - H * prior_Xk_predict);

        // 得到后验协方差
        posterior_Pk_predict = (Eigen::Matrix4f::Identity() - Kk * H) * prior_Pk_predict;
    }
    */

    // 得到先验状态
    Eigen::Vector4f get_prior_Xk_predict()
    {
        return prior_Xk_predict;
    }

    // 得到后验状态
    Eigen::Vector4f get_posterior_Xk_predict()
    {
        return posterior_Xk_predict;
    }
};

class ImageSub : public rclcpp::Node
{
private:
    // 滤波器是否已初始化
    bool is_kalman_initialized = false;

    // 当前帧是否检测到有效装甲板
    bool has_valid_armor = false;

    // 订阅者指针
    rclcpp::Subscription<NNDetectArray>::SharedPtr my_subscriber;

    // 图像
    cv::Mat my_img;

    // 存储所有装甲板
    std::vector<NNDetectData> my_detections;

    // 存储一个装甲板上四个角点
    std::vector<cv::Point3f> my_keypoints;

    // 4个角点的x坐标之和（用于求中心点的x坐标）
    float my_x_sum = 0.0f;

    // 4个角点的y坐标之和（用于求中心点的y坐标）
    float my_y_sum = 0.0f;

    // 中心点
    cv::Point2f my_armor_center;

    // 上一帧时间
    rclcpp::Time my_last_frame_time;

    // 当前帧时间
    rclcpp::Time my_current_frame_time;

    // 上一帧和当前帧的时间间隔
    double my_dt;

    // 卡尔曼滤波器的智能指针
    std::shared_ptr<KalmanFilter2D> my_kalman;

    // 显示窗口是否打开
    bool window_created = false;

public:
    // 每次订阅到数据，都会调用回调函数把图片数据转换为Mat格式
    ImageSub() : Node("image_sub")
    {
        // 初始化上一帧时间
        my_last_frame_time = this->get_clock()->now() - rclcpp::Duration::from_seconds(1.0);

        // 创建订阅者
        my_subscriber = this->create_subscription<NNDetectArray>(
            "detect_results",                                               // 订阅的话题的名字
            10,                                                             // QoS
            std::bind(&ImageSub::sub_callback, this, std::placeholders::_1) // 回调函数，把接受到的图像转成Mat
        );
        RCLCPP_INFO(this->get_logger(), "图像订阅节点已启动，等待消息...");
    }

    // 订阅节点image_sub的回调函数
    void sub_callback(const NNDetectArray::SharedPtr msg)
    {
        my_current_frame_time = msg->header.stamp;
        rclcpp::Duration dt_duration = my_current_frame_time - my_last_frame_time;
        my_dt = dt_duration.seconds(); // 转为秒（如0.083s、0.085s等）
        // 过滤异常dt（避免过小/过大，比如0.001s或1s以上）
        if (my_dt <= 0.001 || my_dt >= 1.0)
        {
            RCLCPP_WARN(this->get_logger(), "无效dt: %.4f s，跳过本次滤波", my_dt);
            my_last_frame_time = my_current_frame_time; // 更新时间戳
            return;
        }
        try
        {
            // 1. 接受msg的图像和装甲板数据
            // 先把上一帧的装甲板数据删掉
            my_detections.clear();
            // 把上一帧的角点数据删掉
            my_keypoints.clear();
            const sensor_msgs::msg::Image &img_msg = msg->image;
            my_detections = msg->detections;

            // 2. 基础尺寸检查（排除负数/零）
            if (img_msg.width <= 0 || img_msg.height <= 0)
            {
                RCLCPP_ERROR(this->get_logger(), "无效尺寸: width=%d, height=%d",
                             img_msg.width, img_msg.height);
                return;
            }

            // 3. 检查 data 长度是否匹配（step * height）
            size_t expected_data_len = img_msg.step * img_msg.height;
            if (img_msg.data.empty() || img_msg.data.size() != expected_data_len)
            {
                RCLCPP_ERROR(this->get_logger(), "数据长度异常: 预期%zu字节, 实际%zu字节",
                             expected_data_len, img_msg.data.size());
                return;
            }

            // 4. 关键：手动构造 OpenCV Mat（绕开 cv_bridge 问题）
            // CV_8UC3 对应 bgr8 编码（8位无符号，3通道）
            // 参数依次：高度(rows)、宽度(cols)、数据类型、数据指针、每行步长
            cv::Mat temp_img(
                img_msg.height,
                img_msg.width,
                CV_8UC3,
                const_cast<unsigned char *>(img_msg.data.data()), // 转换为 OpenCV 兼容的指针
                img_msg.step);

            // 5. 检查构造的 Mat 是否有效（避免空矩阵）
            if (temp_img.empty() ||
                static_cast<unsigned int>(temp_img.rows) != img_msg.height ||
                static_cast<unsigned int>(temp_img.cols) != img_msg.width)
            {
                RCLCPP_ERROR(this->get_logger(), "手动构造 Mat 失败");
                return;
            }

            // 6. 克隆矩阵（避免原数据指针失效，因为 msg 生命周期可能结束）
            my_img = temp_img.clone();
            /*
            RCLCPP_INFO(this->get_logger(), "成功构造图像: %dx%d, 通道数=%d",
                        my_img.cols, my_img.rows, my_img.channels());
            */

            // 7. 在my_img上绘制四个角点与中心点
            draw_original_center();

            // 8. 进行2D卡尔曼滤波，获取先验的中心点坐标和后验中心点坐标
            // -------------------------- 核心：滤波器逻辑 --------------------------
            if (!is_kalman_initialized)
            {
                // 场景1：未初始化 + 有有效装甲板 → 首次创建滤波器
                if (has_valid_armor)
                {
                    my_kalman = std::make_shared<KalmanFilter2D>(my_armor_center, my_dt);
                    is_kalman_initialized = true;
                    RCLCPP_INFO(this->get_logger(), "卡尔曼滤波器首次初始化成功");
                }
                else
                {
                    // 未初始化且无装甲板 → 跳过（无初始状态，无法预测）
                    RCLCPP_WARN(this->get_logger(), "未检测到有效装甲板，暂不初始化滤波器");
                }
            }
            else
            {
                // 场景2/3：已初始化 → 先执行预测（无论有无装甲板，都要演进状态）
                my_kalman->predict(my_dt);

                if (has_valid_armor)
                {
                    // 场景2：有装甲板 → 执行更新（用观测值修正预测）
                    my_kalman->update(my_armor_center);
                }
                else
                {
                    // 场景3：无装甲板 → 仅预测（用历史速度继续推演，避免状态重置）
                    RCLCPP_WARN(this->get_logger(), "当前帧无有效装甲板，仅执行预测");
                }

                // （5. 绘制滤波结果：先验预测（绿色）、后验修正（红色））
                Eigen::Vector4f prior_state = my_kalman->get_prior_Xk_predict();
                Eigen::Vector4f posterior_state = my_kalman->get_posterior_Xk_predict();
                // 先验位置（预测的中心点：prior_state(0)=x, prior_state(2)=y）
                cv::circle(my_img, cv::Point(prior_state(0), prior_state(2)), 5, cv::Scalar(0, 255, 0), -1);
                // 后验位置（修正后的中心点）
                cv::circle(my_img, cv::Point(posterior_state(0), posterior_state(2)), 5, cv::Scalar(0, 0, 255), -1);
            }

            //  显示图像
            if (!window_created)
            {
                cv::namedWindow("sub_img", cv::WINDOW_NORMAL);
                window_created = true;
            }

            cv::imshow("sub_img", my_img);
            if (cv::waitKey(1))
            {
            }

            // 更新上一帧时间戳
            my_last_frame_time = my_current_frame_time;
        }
        catch (const cv::Exception &e)
        {
            RCLCPP_ERROR(this->get_logger(), "OpenCV 错误: %s", e.what());
        }
        catch (const std::exception &e)
        {
            RCLCPP_ERROR(this->get_logger(), "通用错误: %s", e.what());
        }
    }

    // 销毁窗口
    ~ImageSub()
    {
        if (window_created)
        {
            cv::destroyWindow("sub_img");
        }
    }

    // 获取图像
    cv::Mat get_sub_img()
    {
        return my_img;
    }

    // 在当前帧Mat图像绘制中心点（已知msg的四个角点坐标）
    void draw_original_center()
    {
        // 先重置为false，避免上一帧残留
        has_valid_armor = false;

        // 在当前帧，遍历每一个装甲板，先通过置信度筛选
        for (auto &armor : my_detections)
        {
            my_x_sum = 0.0f;
            my_y_sum = 0.0f;
            if (armor.confidence < 0.5 || armor.keypoints.size() != 4)
            {
                continue;
            }
            // 遍历每个装甲板的四个角点
            for (auto &point : armor.keypoints)
            {
                /*
                左上-左下-右下-右上
                像素坐标系，齐次为3维
                点1：(956,460,1)
                点2：(950,525,1)
                点3：(1099,536,1)
                点4：(1104,475,1)

                std::cout << "点" << points_count << "："
                <<"(" << point.x << "," << point.y << "," << point.z << ")"
                << std::endl;
                points_count++;
                */
                // 把四个角点用蓝色绘制出来
                cv::circle(my_img, cv::Point(point.x, point.y), 5, cv::Scalar(255, 0, 0), -1); // 蓝色填充，半径30
                my_x_sum += point.x;
                my_y_sum += point.y;
            }
            my_armor_center.x = my_x_sum / 4;
            my_armor_center.y = my_y_sum / 4;
            cv::circle(my_img, cv::Point(my_armor_center.x, my_armor_center.y), 5, cv::Scalar(255, 0, 0), -1); // 蓝色填充，半径30
            has_valid_armor = true;                                                                            // 标记当前帧有有效装甲板
            break;
        }
    }
};

int main(int argc, char *argv[])
{
    rclcpp::init(argc, argv);
    // 初始化 OpenCV 窗口（提前创建，避免回调中竞争）
    cv::namedWindow("sub_img", cv::WINDOW_NORMAL);
    // 运行订阅节点，获取包资源
    auto node = std::make_shared<ImageSub>();
    rclcpp::spin(node);

    // 清理资源
    rclcpp::shutdown();
    cv::destroyAllWindows();
    return 0;
}