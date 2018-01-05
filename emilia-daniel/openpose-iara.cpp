// ------------------------- OpenPose Library Tutorial - Pose - Example 2 - Extract Pose or Heatmap from Image -------------------------
// This second example shows the user how to:
    // 1. Load an image (`filestream` module)
    // 2. Extract the pose of that image (`pose` module)
    // 3. Render the pose or heatmap on a resized copy of the input image (`pose` module)
    // 4. Display the rendered pose or heatmap (`gui` module)
// In addition to the previous OpenPose modules, we also need to use:
    // 1. `core` module: for the Array<float> class that the `pose` module needs
    // 2. `utilities` module: for the error & logging functions, i.e. op::error & op::log respectively

// 3rdparty dependencies
// GFlags: DEFINE_bool, _int32, _int64, _uint64, _double, _string
#include <gflags/gflags.h>
// Allow Google Flags in Ubuntu 14
#ifndef GFLAGS_GFLAGS_H_
    namespace gflags = google;
#endif
// OpenPose dependencies
#include <openpose/core/headers.hpp>
#include <openpose/filestream/headers.hpp>
#include <openpose/gui/headers.hpp>
#include <openpose/pose/headers.hpp>
#include <openpose/utilities/headers.hpp>
#include <iostream>
#include <fstream>
#include <string>
#include <cmath>
using namespace std;

// See all the available parameter options withe the `--help` flag. E.g. `build/examples/openpose/openpose.bin --help`
// Note: This command will show you flags for other unnecessary 3rdparty files. Check only the flags for the OpenPose
// executable. E.g. for `openpose.bin`, look for `Flags from examples/openpose/openpose.cpp:`.
// Debugging/Other
DEFINE_int32(logging_level,             3,              "The logging level. Integer in the range [0, 255]. 0 will output any log() message, while"
                                                        " 255 will not output any. Current OpenPose library messages are in the range 0-4: 1 for"
                                                        " low priority messages and 4 for important ones.");
// Producer
DEFINE_string(image_path,               "examples/media/COCO_val2014_000000000192.jpg",     "Process the desired image.");
// OpenPose
DEFINE_string(model_pose,               "COCO",         "Model to be used. E.g. `COCO` (18 keypoints), `MPI` (15 keypoints, ~10% faster), "
                                                        "`MPI_4_layers` (15 keypoints, even faster but less accurate).");
DEFINE_string(model_folder,             "models/",      "Folder path (absolute or relative) where the models (pose, face, ...) are located.");
DEFINE_string(net_resolution,           "-1x368",       "Multiples of 16. If it is increased, the accuracy potentially increases. If it is"
                                                        " decreased, the speed increases. For maximum speed-accuracy balance, it should keep the"
                                                        " closest aspect ratio possible to the images or videos to be processed. Using `-1` in"
                                                        " any of the dimensions, OP will choose the optimal aspect ratio depending on the user's"
                                                        " input value. E.g. the default `-1x368` is equivalent to `656x368` in 16:9 resolutions,"
                                                        " e.g. full HD (1980x1080) and HD (1280x720) resolutions.");
DEFINE_string(output_resolution,        "-1x-1",        "The image resolution (display and output). Use \"-1x-1\" to force the program to use the"
                                                        " input image resolution.");
DEFINE_int32(num_gpu_start,             0,              "GPU device start number.");
DEFINE_double(scale_gap,                0.3,            "Scale gap between scales. No effect unless scale_number > 1. Initial scale is always 1."
                                                        " If you want to change the initial scale, you actually want to multiply the"
                                                        " `net_resolution` by your desired initial scale.");
DEFINE_int32(scale_number,              1,              "Number of scales to average.");
// OpenPose Rendering
DEFINE_int32(part_to_show,              0,             "Prediction channel to visualize (default: 0). 0 for all the body parts, 1-18 for each body"
                                                        " part heat map, 19 for the background heat map, 20 for all the body part heat maps"
                                                        " together, 21 for all the PAFs, 22-40 for each body part pair PAF");
DEFINE_bool(disable_blending,           false,          "If enabled, it will render the results (keypoint skeletons or heatmaps) on a black"
                                                        " background, instead of being rendered into the original image. Related: `part_to_show`,"
                                                        " `alpha_pose`, and `alpha_pose`.");
DEFINE_double(render_threshold,         0.05,           "Only estimated keypoints whose score confidences are higher than this threshold will be"
                                                        " rendered. Generally, a high threshold (> 0.5) will only render very clear body parts;"
                                                        " while small thresholds (~0.1) will also output guessed and occluded keypoints, but also"
                                                        " more false positives (i.e. wrong detections).");
DEFINE_double(alpha_pose,               0.6,            "Blending factor (range 0-1) for the body part rendering. 1 will show it completely, 0 will"
                                                        " hide it. Only valid for GPU rendering.");
DEFINE_double(alpha_heatmap,            0.7,            "Blending factor (range 0-1) between heatmap and original frame. 1 will only show the"
                                                        " heatmap, 0 will only show the frame. Only valid for GPU rendering.");

class Frame{
public:
	vector<cv::Rect> GTboundingBoxes; //ground truth bounding boxes
	vector<cv::Rect> PredictionBoundingBoxes; //ground truth bounding boxes
	void appendGTBoundingBox(cv::Rect rect){
		GTboundingBoxes.push_back(rect);
	}
	void appendPredictionBoundingBox(cv::Rect rect){
		PredictionBoundingBoxes.push_back(rect);
	}
};

//compute intersection over union of two bounding boxes
float iou(cv::Rect a, cv::Rect b){
	cv::Rect intersection = a & b;
	cv::Rect u = a | b;

	if (u.width * u.height != 0)
		return ((float)intersection.width * (float)intersection.height) / ((float)u.width * (float)u.height);
	return 0;
}

vector<Frame> loadGroundTruth(){
	cout << "Loading ground truth...\n";
	ifstream in("groundtruth.txt");
	if (!in){
		cout << "Can't open groundtruth.txt\n";
		exit(0);
	}
	int x,y,w,h, frameNumber;
	char comma; //dummy
	vector<Frame> frames; // result
	while (
			(in >> frameNumber) && (in >> comma) &&
			(in >> x) && (in >> comma) &&
			  (in >> y) && (in >> comma) &&
			  (in >> w) && (in >> comma) &&
			  (in >> h)) {


			while (frames.size() < frameNumber)
				frames.push_back(Frame());
			if (frames.size() == frameNumber)
				frames.push_back(Frame());
			frames[frameNumber].appendGTBoundingBox(cv::Rect(x,y,w,h));
	}
	in.close();
	return frames;
};

void drawRect(cv::Mat frame, cv::Rect rect, cv::Scalar color){

	cv::rectangle(
			frame,
			cv::Point(rect.x, rect.y),
			cv::Point(rect.x + rect.width, rect.y + rect.height),
			color, 2, 8, 0);

}

int openPoseTutorialPose2()
{
    op::log("OpenPose Library Tutorial - Example 2.", op::Priority::High);
    // ------------------------- INITIALIZATION -------------------------
    // Step 1 - Set logging level
        // - 0 will output all the logging messages
        // - 255 will output nothing
    op::check(0 <= FLAGS_logging_level && FLAGS_logging_level <= 255, "Wrong logging_level value.",
              __LINE__, __FUNCTION__, __FILE__);
    op::ConfigureLog::setPriorityThreshold((op::Priority)FLAGS_logging_level);
    op::log("", op::Priority::Low, __LINE__, __FUNCTION__, __FILE__);
    // Step 2 - Read Google flags (user defined configuration)
    // outputSize
    const auto outputSize = op::flagsToPoint(FLAGS_output_resolution, "-1x-1");
    // netInputSize
    const auto netInputSize = op::flagsToPoint(FLAGS_net_resolution, "-1x368");
    // poseModel
    const auto poseModel = op::flagsToPoseModel(FLAGS_model_pose);
    // Check no contradictory flags enabled
    if (FLAGS_alpha_pose < 0. || FLAGS_alpha_pose > 1.)
        op::error("Alpha value for blending must be in the range [0,1].", __LINE__, __FUNCTION__, __FILE__);
    if (FLAGS_scale_gap <= 0. && FLAGS_scale_number > 1)
        op::error("Incompatible flag configuration: scale_gap must be greater than 0 or scale_number = 1.",
                  __LINE__, __FUNCTION__, __FILE__);
    // Enabling Google Logging
    const bool enableGoogleLogging = true;
    // Logging
    op::log("", op::Priority::Low, __LINE__, __FUNCTION__, __FILE__);
    // Step 3 - Initialize all required classes
    op::ScaleAndSizeExtractor scaleAndSizeExtractor(netInputSize, outputSize, FLAGS_scale_number, FLAGS_scale_gap);
    op::CvMatToOpInput cvMatToOpInput;
    op::CvMatToOpOutput cvMatToOpOutput;
    auto poseExtractorPtr = std::make_shared<op::PoseExtractorCaffe>(
        poseModel, FLAGS_model_folder, FLAGS_num_gpu_start, std::vector<op::HeatMapType>{}, op::ScaleMode::ZeroToOne,
        enableGoogleLogging
    );
    op::PoseGpuRenderer poseGpuRenderer{poseModel, poseExtractorPtr, (float)FLAGS_render_threshold,
                                        !FLAGS_disable_blending, (float)FLAGS_alpha_pose, (float)FLAGS_alpha_heatmap};
    poseGpuRenderer.setElementToRender(FLAGS_part_to_show);
    op::OpOutputToCvMat opOutputToCvMat;
    // Step 4 - Initialize resources on desired thread (in this case single thread, i.e. we init resources here)
    poseExtractorPtr->initializationOnThread();
    poseGpuRenderer.initializationOnThread();

    // ------------------------- POSE ESTIMATION AND RENDERING -------------------------
    // Step 1 - Read and load image, error if empty (possibly wrong path)
    // Alternative: cv::imread(FLAGS_image_path, CV_LOAD_IMAGE_COLOR);


    cv::VideoCapture cap("output.mp4"); // open the default camera
	if(!cap.isOpened())  // check if we succeeded
		return -1;
	cv::namedWindow("OpenPose",1);
	double fps = cap.get(CV_CAP_PROP_FPS);
	printf("Video Frame Rate: %f\n", fps);

	int frameNumber = 0;
	vector <Frame> frames = loadGroundTruth();
	vector <float> iouList;

	for(;;)
	{

		cv::Mat inputImage;
		cap >> inputImage; // get a new frame from camera

		if(inputImage.empty()){
			break;
		}
		const op::Point<int> imageSize{inputImage.cols, inputImage.rows};
		// Step 2 - Get desired scale sizes
		std::vector<double> scaleInputToNetInputs;
		std::vector<op::Point<int>> netInputSizes;
		double scaleInputToOutput;
		op::Point<int> outputResolution;
		std::tie(scaleInputToNetInputs, netInputSizes, scaleInputToOutput, outputResolution)
			= scaleAndSizeExtractor.extract(imageSize);
		// Step 3 - Format input image to OpenPose input and output formats
		const auto netInputArray = cvMatToOpInput.createArray(inputImage, scaleInputToNetInputs, netInputSizes);
		auto outputArray = cvMatToOpOutput.createArray(inputImage, scaleInputToOutput, outputResolution);
		// Step 4 - Estimate poseKeypoints
		poseExtractorPtr->forwardPass(netInputArray, imageSize, scaleInputToNetInputs);
		const auto poseKeypoints = poseExtractorPtr->getPoseKeypoints();

		string strFrameNumber = "Frame number: " + std::to_string(frameNumber);
		cv::putText(inputImage,  strFrameNumber, cv::Point(40,40), CV_FONT_HERSHEY_PLAIN, 1.2, cv::Scalar(0,255,0), 1, CV_AA);

		string p = "PREDICTION";
		string gt = "GROUND TRUTH";
		cv::putText(inputImage,  p, cv::Point(40,60), CV_FONT_HERSHEY_PLAIN, 1.2, cv::Scalar(0,0,255), 1, CV_AA);
		cv::putText(inputImage,  gt, cv::Point(40,80), CV_FONT_HERSHEY_PLAIN, 1.2, cv::Scalar(255,0,0), 1, CV_AA);

		for (auto person = 0; person < poseKeypoints.getSize(0) ; person++){

			const auto thresholdRectangle = 0.1f;
			const auto numberPersons = poseKeypoints.getSize(0);
			const auto numberKeypoints = poseKeypoints.getSize(1);
			const auto areaKeypoints = numberKeypoints * poseKeypoints.getSize(2);
			const auto personRectangle = op::getKeypointsRectangle(poseKeypoints, person, numberKeypoints, thresholdRectangle);

			cv::Rect predictedBoundingBox = cv::Rect(
					cv::Point(personRectangle.x, personRectangle.y),
					cv::Point(personRectangle.x + personRectangle.width,personRectangle.y + personRectangle.height));

			//draw predicted boundingbox
			drawRect(inputImage, predictedBoundingBox, cv::Scalar(0,0,255));


			for (int i = 0; i < frames[frameNumber].GTboundingBoxes.size(); i++){

				//compute intersection over union and append on list
				iouList.push_back(iou(predictedBoundingBox, frames[frameNumber].GTboundingBoxes[i]));

				//draw ground truth boundingbox
				drawRect(inputImage, frames[frameNumber].GTboundingBoxes[i], cv::Scalar(255,0,0));
			}



			frames[frameNumber].appendPredictionBoundingBox(predictedBoundingBox);

			cv::imshow("OpenPose", inputImage);
		}

		frameNumber++;
		if (cv::waitKey(30) >= 0) break;
	}

	float average = accumulate( iouList.begin(), iouList.end(), 0.0)/iouList.size();
	float standardDeviation = 0.0;
	for(int i = 0; i < iouList.size(); ++i)
		standardDeviation += pow(iouList[i] - average, 2);

	cout << "The average of iou is : " << average << endl;
	cout << "The standardDeviation of iou is : " << standardDeviation << endl;

    return 0;
}

int main(int argc, char *argv[])
{
    // Parsing command line flags
    gflags::ParseCommandLineFlags(&argc, &argv, true);

    // Running openPoseTutorialPose2
    return openPoseTutorialPose2();
}
