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
public:
    ImageSub(/* args */) : Node("image_sub")
    {
        my_subscriber = this->create_subscription<NNDetectArray>("detect_results", 10, [&](const NNDetectArray::SharedPtr msg) -> void
        { 
            /*
            detect_bag1包作为发布者，发布了话题/detect_results
            该话题的消息接口类型是detect_msg/msg/NNDetectArray
            ros2 topic info /detect_results 
            Type: detect_msg/msg/NNDetectArray
            Publisher count: 1
            */
            //回调函数，每次接收到图像数据要干的事

            //遍历每一个NNDetectData
            // for (const auto &armor : (msg->detections))
            // {
            //     if(armor.confidence < 0.5)
            //     {
            //         continue;
            //     }
            // }
            cv_bridge::CvImagePtr cv_ptr = cv_bridge::toCvCopy(msg->image,"bgr8");
            my_img = cv_ptr->image; 
            cv::imshow("sub_img",my_img);
            cv::waitKey(1);
        });
        
    }

    cv::Mat get_sub_img()
    {
        return my_img;
    }

    
};

int main(int argc, char *argv[])
{
    
    rclcpp::init(argc, argv);

    auto node = std::make_shared<ImageSub>();
    rclcpp::spin(node);  //阻塞
    
    
    

    return 0;
}