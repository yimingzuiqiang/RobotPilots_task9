#include <detect_msg/msg/nn_detect_data.hpp>
#include <detect_msg/msg/nn_detect_array.hpp>
#include <opencv2/opencv.hpp>
#include <rclcpp/rclcpp.hpp>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/msg/image.hpp>

using NNDetectData = detect_msg::msg::NNDetectData;
using NNDetectArray = detect_msg::msg::NNDetectArray;

class ImageSub : public rclcpp::Node
{
private:
    rclcpp::Subscription<NNDetectArray>::SharedPtr my_subscriber;
    cv::Mat my_img;
    bool window_created = false;

public:
    ImageSub() : Node("image_sub")
    {
        my_subscriber = this->create_subscription<NNDetectArray>(
            "detect_results",
            10,
            [this](const NNDetectArray::SharedPtr msg)
            {
                try
                {
                    // 仅保留一段处理逻辑，集中做参数检查和图像构造
                    const sensor_msgs::msg::Image &img_msg = msg->image;

                    // 1. 基础尺寸检查（排除负数/零）
                    if (img_msg.width <= 0 || img_msg.height <= 0)
                    {
                        RCLCPP_ERROR(this->get_logger(), "无效尺寸: width=%d, height=%d",
                                     img_msg.width, img_msg.height);
                        return;
                    }

                    // 2. 检查 data 长度是否匹配（step * height）
                    size_t expected_data_len = img_msg.step * img_msg.height;
                    if (img_msg.data.empty() || img_msg.data.size() != expected_data_len)
                    {
                        RCLCPP_ERROR(this->get_logger(), "数据长度异常: 预期%d字节, 实际%d字节",
                                     expected_data_len, img_msg.data.size());
                        return;
                    }

                    // 3. 关键：手动构造 OpenCV Mat（绕开 cv_bridge 问题）
                    // CV_8UC3 对应 bgr8 编码（8位无符号，3通道）
                    // 参数依次：高度(rows)、宽度(cols)、数据类型、数据指针、每行步长
                    cv::Mat temp_img(
                        img_msg.height,
                        img_msg.width,
                        CV_8UC3,
                        const_cast<unsigned char *>(img_msg.data.data()), // 转换为 OpenCV 兼容的指针
                        img_msg.step);

                    // 4. 检查构造的 Mat 是否有效（避免空矩阵）
                    if (temp_img.empty() || temp_img.rows != img_msg.height || temp_img.cols != img_msg.width)
                    {
                        RCLCPP_ERROR(this->get_logger(), "手动构造 Mat 失败");
                        return;
                    }

                    // 5. 克隆矩阵（避免原数据指针失效，因为 msg 生命周期可能结束）
                    my_img = temp_img.clone();
                    RCLCPP_INFO(this->get_logger(), "成功构造图像: %dx%d, 通道数=%d",
                                my_img.cols, my_img.rows, my_img.channels());

                    // 6. 显示图像
                    if (!window_created)
                    {
                        cv::namedWindow("sub_img", cv::WINDOW_NORMAL);
                        window_created = true;
                    }
                    cv::imshow("sub_img", my_img);
                    cv::waitKey(1); // 必须调用，否则窗口无响应
                }
                catch (const cv::Exception &e)
                {
                    RCLCPP_ERROR(this->get_logger(), "OpenCV 错误: %s", e.what());
                }
                catch (const std::exception &e)
                {
                    RCLCPP_ERROR(this->get_logger(), "通用错误: %s", e.what());
                }
            });
        RCLCPP_INFO(this->get_logger(), "图像订阅节点已启动，等待消息...");
    }

    ~ImageSub()
    {
        if (window_created)
        {
            cv::destroyWindow("sub_img");
        }
    }

    cv::Mat get_sub_img()
    {
        return my_img;
    }
};

int main(int argc, char *argv[])
{
    rclcpp::init(argc, argv);
    // 初始化 OpenCV 窗口（提前创建，避免回调中竞争）
    cv::namedWindow("sub_img", cv::WINDOW_NORMAL);
    auto node = std::make_shared<ImageSub>();
    rclcpp::spin(node);
    // 清理资源
    rclcpp::shutdown();
    cv::destroyAllWindows();
    return 0;
}