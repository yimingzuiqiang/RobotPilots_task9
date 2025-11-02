#include <detect_msg/msg/nn_detect_data.hpp>
#include <detect_msg/msg/nn_detect_array.hpp>
#include <opencv2/opencv.hpp>
#include <rclcpp/rclcpp.hpp>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/msg/image.hpp>
#include <vector>

using NNDetectData = detect_msg::msg::NNDetectData;
using NNDetectArray = detect_msg::msg::NNDetectArray;

class ImageSub : public rclcpp::Node
{
private:
    rclcpp::Subscription<NNDetectArray>::SharedPtr my_subscriber;
    cv::Mat my_img;
    // 存储所有装甲板
    std::vector<NNDetectData> my_detections;
    // 存储一个装甲板上四个角点
    std::vector<cv::Point3f> my_keypoints;
    bool window_created = false;

public:
    // 每次订阅到数据，都会调用回调函数把图片数据转换为Mat格式
    ImageSub() : Node("image_sub")
    {
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
                RCLCPP_ERROR(this->get_logger(), "数据长度异常: 预期%d字节, 实际%d字节",
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
            if (temp_img.empty() || temp_img.rows != img_msg.height || temp_img.cols != img_msg.width)
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

            // 7. 在my_img上绘制四个角点
            draw_original_center();

            //  显示图像
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
        // 在当前帧，遍历每一个装甲板，先通过置信度筛选
        for (auto &armor : my_detections)
        {
            if (armor.confidence < 0.5)
            {
                continue;
            }

            // 绘制四个角点(红色)
            // 先输出4个角点坐标
            // std::cout << "装甲板点的数量：" << armor.keypoints.size() << std::endl;

            // int points_count = 1;
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
               //把四个角点用蓝色绘制出来
                cv::circle(my_img, cv::Point(point.x, point.y), 5, cv::Scalar(255, 0, 0), -1); // 黄色填充，半径30
            }
            // std::cout << std::endl;
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