#include "controller/yolov7controller.h"

int main(int argc, char *argv[]) {
    // Get the project path
    std::string filePath = __FILE__;
    std::string ProjectPath = filePath.substr(0, filePath.find_last_of("/\\"));

    // Define paths
    std::string lpCharacterModel = ProjectPath + "/weights/lp_character.engine";
    std::string lpCharacterClass = ProjectPath + "/classes/character.names";
    std::string imagePath = ProjectPath + "/yolov7/images/license_plate/1.jpg";
    std::string outputPath = ProjectPath + "/output.jpg";

    // Initialize the model
    ObjectDetection *lpCharacter = new ObjectDetection(lpCharacterModel, lpCharacterClass);

    // Detect the image
    cv::Mat image = cv::imread(imagePath);
    image = lpCharacter->detect(image);

    // Save the output image
    cv::imwrite(outputPath, image);
    std::cout << "Output saved to: " << outputPath << std::endl;

    delete lpCharacter;

    return 0;
}