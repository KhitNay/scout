#include <rclcpp/rclcpp.hpp>
#include <realsense2_camera_msgs/msg/rgbd.hpp>
#include <cv_bridge/cv_bridge.hpp>

#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <onnxruntime_cxx_api.h>

#include <memory>

struct Detection
{
    cv::Rect2f box;
    float confidence;
};

class DetectionNode : public rclcpp::Node
{
    public:
        DetectionNode()
            : Node("scout_detection")
        {
            // Set up ROS interfaces
            mRGBDSub = create_subscription<realsense2_camera_msgs::msg::RGBD>("/camera/camera/rgbd", 10,
                            std::bind(&DetectionNode::rgbdCallback, this, std::placeholders::_1));
            mCameraInfoSub = create_subscription<sensor_msgs::msg::CameraInfo>("/camera/camera/color/camera_info", 10,
                            std::bind(&DetectionNode::cameraInfoCallback, this, std::placeholders::_1));
            mImagePub = create_publisher<sensor_msgs::msg::Image>("/scout/detection_image", 10);

            // ONNX Runtime setup
            initOnnxRuntime();

            RCLCPP_INFO(get_logger(), "Scout Detection Node Started");
        }

    private:

        void rgbdCallback(const realsense2_camera_msgs::msg::RGBD::SharedPtr rgbd)
        {
            if (!mIntrinsicsReceived)
            {
                RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 1000, "Intrinsics not received");
                return;
            }

            cv::Mat bgrFrame;
            cv::Mat depthFrame;
            if (!extractFrames(rgbd, bgrFrame, depthFrame))
            {
                return;
            }

            std::vector<float> inputData = preprocess(bgrFrame);
            std::array<int64_t, 4> inputShape = {1, 3, MODEL_INPUT_SIZE, MODEL_INPUT_SIZE};

            Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
            Ort::Value inputTensor = Ort::Value::CreateTensor<float>(
                memoryInfo,
                inputData.data(),
                inputData.size(),
                inputShape.data(),
                inputShape.size()
            );

            const char* inputNames[]  = {"images"};
            const char* outputNames[] = {"output0"};

            std::vector<Ort::Value> outputTensors;
            try
            {
                outputTensors = mOrtSession->Run(
                    Ort::RunOptions{nullptr},
                    inputNames,
                    &inputTensor,
                    1,
                    outputNames,
                    1
                );
            }
            catch (const Ort::Exception& e)
            {
                RCLCPP_ERROR(get_logger(), "Ort exception: %s", e.what());
                return;
            }

            const float* rawOutput = outputTensors[0].GetTensorData<float>();
            std::vector<Detection> detections = postprocess(rawOutput, bgrFrame.cols, bgrFrame.rows);
            cv::Mat annotatedFrame = bgrFrame.clone();

            for (const Detection& detection : detections)
            {
                // Get centre of box
                int u = static_cast<int>(detection.box.x + detection.box.width/2);
                int v = static_cast<int>(detection.box.y + detection.box.height/2);

                u = std::clamp(u, 0, bgrFrame.cols - 1);
                v = std::clamp(v, 0, bgrFrame.rows - 1);

                // Check for valid depth
                uint16_t depth = depthFrame.at<uint16_t>(v, u);
                if (!depth)
                {
                    RCLCPP_WARN(get_logger(), "Invalid depth. Skipping");
                    continue;
                }

                // 3D projection
                double Z = depth/1000.0;
                double X = (u - mCx) * Z / mFx;
                double Y = (v - mCy) * Z / mFy;

                // Draw detection and label
                cv::Rect boundingBox;
                boundingBox.x = detection.box.x;
                boundingBox.y = detection.box.y;
                boundingBox.width = detection.box.width;
                boundingBox.height = detection.box.height;

                cv::rectangle(annotatedFrame, boundingBox, cv::Scalar(0, 255, 0), 2);
                std::string label = cv::format("Person %.0f%% X:%.1f Y:%.1f Z:%.1f m", detection.confidence * 100.0f, X, Y, Z);
                cv::putText(annotatedFrame, label, cv::Point(boundingBox.x, boundingBox.y - 5),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1);

                RCLCPP_INFO(get_logger(), "Person %.0f%% X:%.1f Y:%.1f Z:%.1f m", detection.confidence * 100.0f, X, Y, Z);
            }

            // Convert annotated image to ROS image and publish
            auto annotatedImage = cv_bridge::CvImage(rgbd->rgb.header, "bgr8", annotatedFrame).toImageMsg();
            mImagePub->publish(*annotatedImage);
        }

        bool extractFrames(const realsense2_camera_msgs::msg::RGBD::SharedPtr& rgbd, 
            cv::Mat& bgrFrame, cv::Mat& depthFrame)
        {
            try
            {
                bgrFrame   = cv_bridge::toCvCopy(rgbd->rgb, sensor_msgs::image_encodings::BGR8)->image;
                depthFrame = cv_bridge::toCvCopy(rgbd->depth, sensor_msgs::image_encodings::TYPE_16UC1)->image;
                return true;
            }
            catch (const cv_bridge::Exception& e)
            {
                RCLCPP_ERROR(get_logger(), "CvBridge exception: %s", e.what()); 
            }
            return false;
        }

        void initOnnxRuntime()
        {
            // Create an environment and session
            mOrtEnv = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "scout_ort_env");

            Ort::SessionOptions sessionOptions;
            sessionOptions.SetInterOpNumThreads(1);
            sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

            const char* modelPath = "/home/khit/scout_ws/src/yolov8n.onnx";
            mOrtSession = std::make_unique<Ort::Session>(*mOrtEnv, modelPath, sessionOptions);

            RCLCPP_INFO(get_logger(), "ONNX Model loaded successfully");
        }

        std::vector<float> preprocess(const cv::Mat& bgrImage)
        {
            // 1. Resize to 640 x 640
            cv::Mat resizedBGR;
            cv::resize(bgrImage, resizedBGR, cv::Size(MODEL_INPUT_SIZE, MODEL_INPUT_SIZE), 0, 0, cv::INTER_LINEAR);

            // 2. BGR -> RGB
            cv::Mat rgbImage;
            cv::cvtColor(resizedBGR, rgbImage, cv::COLOR_BGR2RGB, 0);

            // 3. Normalize to [0, 1]
            cv::Mat normRGB;
            rgbImage.convertTo(normRGB, CV_32F, 1.0/255.0);

            // 4. HWC -> CHW
            std::vector<cv::Mat> channels(normRGB.channels());
            cv::split(normRGB, channels);
            std::vector<float> inputTensor;
            inputTensor.reserve(normRGB.channels() * normRGB.rows * normRGB.cols);

            for (const cv::Mat& channel : channels)
            {
                inputTensor.insert(inputTensor.end(), (float*)channel.datastart, (float*)channel.dataend);
            }
            
            return inputTensor;
        }

        /* Output of YOLOv8 for a single image is a [1, 84, 8400] tensor
        8400 candidates with [cx, cy, width, height] and confidence scores for 80 classes
        The tensor is laid out in memory row-first. All 8400 values for row 0 and so on
        Person class is row 4 */
        std::vector<Detection> postprocess(const float* rawOutput, int originalWidth, int originalHeight)
        {            
            std::vector<Detection> detections;
            std::vector<int> indices;
            std::vector<cv::Rect2d> boxes;
            std::vector<float> scores;

            float scaleX = static_cast<float>(originalWidth)  / MODEL_INPUT_SIZE;
            float scaleY = static_cast<float>(originalHeight) / MODEL_INPUT_SIZE;
            
            // Go through all candidates
            for (int c = 0; c < N_YOLO_CANDIDATES; c++)
            {
                // Discard candidates not meeting threshold
                float personScore = rawOutput[PERSON_CLASS_IDX*N_YOLO_CANDIDATES+c];
                if (personScore < PERSON_THRESHOLD)
                {
                    continue;
                }

                // Convert box format and scale back to original image size
                float cx     = rawOutput[0*N_YOLO_CANDIDATES + c];
                float cy     = rawOutput[1*N_YOLO_CANDIDATES + c];
                float width  = rawOutput[2*N_YOLO_CANDIDATES + c];
                float height = rawOutput[3*N_YOLO_CANDIDATES + c];

                cv::Rect2f box;
                box.x = (cx - width/2) * scaleX;
                box.y = (cy - height/2) * scaleY;
                box.width = width * scaleX;
                box.height = height * scaleY;

                boxes.push_back(box);
                scores.push_back(personScore);
            }

            // NMS
            cv::dnn::NMSBoxes(boxes, scores, PERSON_THRESHOLD, NMS_IOU_THRESHOLD, indices);

            for (auto idx : indices)
            {
                Detection detection;
                detection.box = boxes[idx];
                detection.confidence = scores[idx];
                detections.push_back(detection);
            }

            return detections;
        }

        void cameraInfoCallback(const sensor_msgs::msg::CameraInfo::SharedPtr cameraInfo)
        {
            if (mIntrinsicsReceived)
            {
                return;
            }

            mFx = cameraInfo->k[0];
            mFy = cameraInfo->k[4];
            mCx = cameraInfo->k[2];
            mCy = cameraInfo->k[5];
            mIntrinsicsReceived = true;
            RCLCPP_INFO(get_logger(), "Intrinsics: fx=%.2f fy=%.2f cx=%.2f cy=%.2f (image %dx%d)", 
                mFx, mFy, mCx, mCy, cameraInfo->width, cameraInfo->height);
        } 

        //----------------------------------------------------------------------------------------

        rclcpp::Subscription<realsense2_camera_msgs::msg::RGBD>::SharedPtr mRGBDSub;
        rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr mCameraInfoSub;
        rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr mImagePub;

        std::unique_ptr<Ort::Env> mOrtEnv;
        std::unique_ptr<Ort::Session> mOrtSession;

        static constexpr int   MODEL_INPUT_SIZE  = 640;
        static constexpr int   N_YOLO_CANDIDATES = 8400;
        static constexpr int   PERSON_CLASS_IDX  = 4;
        static constexpr float PERSON_THRESHOLD  = 0.5f;
        static constexpr float NMS_IOU_THRESHOLD = 0.45;

        bool mIntrinsicsReceived = false;
        double mFx;
        double mFy;
        double mCx;
        double mCy;
};

int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    std::shared_ptr<DetectionNode> node = std::make_shared<DetectionNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
}