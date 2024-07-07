// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "Pipeline.h"

Detector::Detector(const std::string &modelDir, const std::string &labelPath,
                   const int cpuThreadNum, const std::string &cpuPowerMode,
                   int inputWidth, int inputHeight,
                   const std::vector<float> &inputMean,
                   const std::vector<float> &inputStd, float scoreThreshold)
    : inputWidth_(inputWidth), inputHeight_(inputHeight), inputMean_(inputMean),
      inputStd_(inputStd), scoreThreshold_(scoreThreshold) {
  paddle::lite_api::MobileConfig config;
  config.set_model_from_file(modelDir + "/model.nb");
  config.set_threads(cpuThreadNum);
  config.set_power_mode(ParsePowerMode(cpuPowerMode));

  predictor_ =
      paddle::lite_api::CreatePaddlePredictor<paddle::lite_api::MobileConfig>(
          config);
  labelList_ = LoadLabelList(labelPath);
  colorMap_ = GenerateColorMap(labelList_.size());

}

std::vector<std::string> Detector::LoadLabelList(const std::string &labelPath) {
  std::ifstream file;
  std::vector<std::string> labels;
  file.open(labelPath);
  while (file) {
    std::string line;
    std::getline(file, line);
    labels.push_back(line);
  }
  file.clear();
  file.close();
  return labels;
}

std::vector<cv::Scalar> Detector::GenerateColorMap(int numOfClasses) {
  std::vector<cv::Scalar> colorMap = std::vector<cv::Scalar>(numOfClasses);
  for (int i = 0; i < numOfClasses; i++) {
    int j = 0;
    int label = i;
    int R = 0, G = 0, B = 0;
    while (label) {
      R |= (((label >> 0) & 1) << (7 - j));
      G |= (((label >> 1) & 1) << (7 - j));
      B |= (((label >> 2) & 1) << (7 - j));
      j++;
      label >>= 3;
    }
    colorMap[i] = cv::Scalar(R, G, B);
  }
  return colorMap;
}

void Detector::Preprocess(const cv::Mat &rgbaImage) {
  // Set the data of input image
  auto inputTensor = predictor_->GetInput(0);
  std::vector<int64_t> inputShape = {1, 3, inputHeight_, inputWidth_};
  inputTensor->Resize(inputShape);
  auto inputData = inputTensor->mutable_data<float>();
  cv::Mat resizedRGBAImage;
  cv::resize(rgbaImage, resizedRGBAImage,
          cv::Size(inputShape[3], inputShape[2]));
  cv::Mat resizedRGBImage;
  cv::cvtColor(resizedRGBAImage, resizedRGBImage, cv::COLOR_BGRA2RGB);
  resizedRGBImage.convertTo(resizedRGBImage, CV_32FC3, 1.0 / 255.0f);
  NHWC3ToNC3HW(reinterpret_cast<const float *>(resizedRGBImage.data), inputData,
               inputMean_.data(), inputStd_.data(), inputShape[3],
               inputShape[2]);
  // Set the size of input image
  auto sizeTensor = predictor_->GetInput(1);
  sizeTensor->Resize({1, 2});
  auto sizeData = sizeTensor->mutable_data<int32_t>();
  sizeData[0] = inputShape[3];
  sizeData[1] = inputShape[2];
}

void Detector::Postprocess(std::vector<RESULT> *results) {
  auto outputTensor = predictor_->GetOutput(0);
  auto outputData = outputTensor->data<float>();
  auto outputShape = outputTensor->shape();
  int outputSize = ShapeProduction(outputShape);
  for (int i = 0; i < outputSize; i += 6) {
    // Class id
    auto class_id = static_cast<int>(round(outputData[i]));
    // Confidence score
    auto score = outputData[i + 1];
    if (score < scoreThreshold_)
      continue;
    RESULT object;
    object.class_name = class_id >= 0 && class_id < labelList_.size()
                            ? labelList_[class_id]
                            : "Unknow";
    object.fill_color = class_id >= 0 && class_id < colorMap_.size()
                            ? colorMap_[class_id]
                            : cv::Scalar(0, 0, 0);
    object.score = score;
    object.x = outputData[i + 2] / inputWidth_;
    object.y = outputData[i + 3] / inputHeight_;
    object.w = (outputData[i + 4] - outputData[i + 2] + 1) / inputWidth_;
    object.h = (outputData[i + 5] - outputData[i + 3] + 1) / inputHeight_;
    results->push_back(object);
  }
}

void Detector::Predict(const cv::Mat &rgbaImage, std::vector<RESULT> *results,
                       double *preprocessTime, double *predictTime,
                       double *postprocessTime) {
  auto t = GetCurrentTime();

  t = GetCurrentTime();
  Preprocess(rgbaImage);
  *preprocessTime = GetElapsedTime(t);
  LOGD("Detector postprocess costs %f ms", *preprocessTime);

  t = GetCurrentTime();
  predictor_->Run();
  *predictTime = GetElapsedTime(t);
  LOGD("Detector predict costs %f ms", *predictTime);

  t = GetCurrentTime();
  Postprocess(results);
  *postprocessTime = GetElapsedTime(t);
  LOGD("Detector postprocess costs %f ms", *postprocessTime);
}


Fom::Fom(const std::string &fommodelDir,
         const int cpuThreadNum, const std::string &cpuPowerMode){
  paddle::lite_api::MobileConfig kp_config;
  kp_config.set_model_from_file(fommodelDir+"/kp_detector_fp32.nb");
  kp_config.set_threads(cpuThreadNum);
  kp_config.set_power_mode(ParsePowerMode(cpuPowerMode));

  kp_predictor_ =
          paddle::lite_api::CreatePaddlePredictor<paddle::lite_api::MobileConfig>(
                  kp_config);

  paddle::lite_api::MobileConfig generator_config;
  generator_config.set_model_from_file(fommodelDir+"/generator_fp32.nb");
  generator_config.set_threads(cpuThreadNum);
  generator_config.set_power_mode(ParsePowerMode(cpuPowerMode));
  generator_predictor_ =
          paddle::lite_api::CreatePaddlePredictor<paddle::lite_api::MobileConfig>(
                  generator_config);
}

void Fom::DrivingProcess() {
  cv::VideoCapture capture;
  std::vector<cv::Mat> driving_video;
  cv::Mat frame;
  capture.open(videoPath_);
  if(!capture.isOpened())
  {
    printf("can not open ...\n");
    return;
  }
  int i = 0;

  while (true)
  {
    //LOGD("size, %d, %d\n",frame.rows, frame.cols);
    bool res = capture.read(frame);
    if (!res) break;
    LOGD("size, %d, %d", i, frame.type());

    KpPreprocess(frame);
    auto t = GetCurrentTime();
    kp_predictor_->Run();
    auto predictKpTime = GetElapsedTime(t);
    LOGD("Detector predict costs %f ms", predictKpTime);
    auto jacobianTensor = kp_predictor_->GetOutput(0);
    auto jacobianData = jacobianTensor->data<float>();
    auto valueTensor = kp_predictor_->GetOutput(1);
    auto valueData = valueTensor->data<float>();

    float *jacobian = new float[40];
    memcpy(jacobian, jacobianData, sizeof(float) * 40);
    float *value = new float[20];
    memcpy(value, valueData, sizeof(float) * 20);
    driving_jacobians.push_back(jacobian);
    driving_values.push_back(value);

    i++;
    //if (i >= 12) break;

    //break;
  }

  capture.release();
  driving_num = driving_jacobians.size();
}

void Fom::KpPreprocess(const cv::Mat &source) {
  // Set the data of input image
  auto inputTensor = kp_predictor_->GetInput(0);
  std::vector<int64_t> inputShape = {1, 3, 256, 256};
  inputTensor->Resize(inputShape);
  auto inputData = inputTensor->mutable_data<float>();


  cv::Mat resizedSource = source.clone();
  cv::resize(source, resizedSource,
             cv::Size(inputShape[3], inputShape[2]));
  cv::Mat resizedRGBSource;
  cv::cvtColor(resizedSource, resizedRGBSource, cv::COLOR_BGR2RGB);

  resizedRGBSource.convertTo(resizedRGBSource, CV_32FC3, 1.0 / 255.0f);
  float inputMean_[3] = {0, 0, 0};
  float inputStd_[3] = {1, 1, 1};
  NHWC3ToNC3HW(reinterpret_cast<const float *>(resizedRGBSource.data), inputData,
               inputMean_, inputStd_, inputShape[3],
               inputShape[2], false);

}


void Fom::Predict(const cv::Mat &image, std::string videoPath,
                  std::vector<cv::Mat> &results,
             double *predictKpTime, double *predictGenTime) {
  auto t = GetCurrentTime();
  //DrivingProcess();
  t = GetCurrentTime();
  cv::Mat resizedRGBSource;
  //cv::cvtColor(image, resizedRGBSource, cv::COLOR_BGR2RGB);
  KpPreprocess(image);
  float preprocessTime = 0;
  preprocessTime = GetElapsedTime(t);
  LOGD("Detector preprocess costs %f ms", preprocessTime);

  t = GetCurrentTime();
  kp_predictor_->Run();
  *predictKpTime = GetElapsedTime(t);
  LOGD("Detector predict costs %f ms", *predictKpTime);

  t = GetCurrentTime();
  auto jacobianTensor = kp_predictor_->GetOutput(0);

    auto jacobianData = jacobianTensor->data<float>();
    for (int i = 0; i < 40; i++) {
        LOGD("pixel %f",  jacobianData[i]);
    }

    auto jacobianShape = jacobianTensor->shape();
  auto valueTensor = kp_predictor_->GetOutput(1);
  auto valueData = valueTensor->data<float>();
  auto valueShape = valueTensor->shape();
  auto sourceTensor = generator_predictor_->GetInput(0);
  std::vector<int64_t> inputShape = {1, 3, 256, 256};
  sourceTensor->Resize(inputShape);
  auto* sourceImageTensor = sourceTensor->mutable_data<float>();
  //sourceTensor = std::move(kp_predictor_->GetInput(0));
  memcpy(sourceImageTensor, kp_predictor_->GetInput(0)->mutable_data<float>(), 3*256*256*sizeof(float));
  //sourceTensor->Resize(inputShape);
  //auto sourceData = sourceTensor->mutable_data<float>();
  auto sourceJacobian = generator_predictor_->GetInput(1);
  sourceJacobian->Resize(jacobianShape);
  auto* data = sourceJacobian->mutable_data<float>();
  size_t jacobianShapeNum = jacobianShape[0]*jacobianShape[1]*jacobianShape[2]*jacobianShape[3];//*sizeof(float);
  //memcpy(data, jacobianData, jacobianShapeNum);
  auto sourceValue = generator_predictor_->GetInput(2);
  sourceValue->Resize(valueShape);
  auto* sourceValueData = sourceValue->mutable_data<float>();
  size_t valueShapeNum = valueShape[0]*valueShape[1]*valueShape[2];//*sizeof(float);
  memcpy(sourceValueData, valueData, valueShapeNum * sizeof(float));
  memcpy(data, jacobianData, jacobianShapeNum * sizeof(float));

  videoPath_ = videoPath;
  DrivingProcess();
  auto drivingInitJacobian = generator_predictor_->GetInput(5);
  drivingInitJacobian->Resize(jacobianShape);
  LOGD("jacobian_value shape");
  auto* drivingInitJacobianData = drivingInitJacobian->mutable_data<float>();
  //memcpy(drivingInitJacobianData, driving_jacobians[0], jacobianShapeNum);
  memcpy(drivingInitJacobianData, driving_jacobians[0], jacobianShapeNum * sizeof(float));
  auto drivingInitValue = generator_predictor_->GetInput(6);
  drivingInitValue->Resize(valueShape);
  auto* drivingInitValueData = drivingInitValue->mutable_data<float>();
  memcpy(drivingInitValueData, driving_values[0], valueShapeNum * sizeof(float));
  auto drivingJacobian = generator_predictor_->GetInput(3);
  drivingJacobian->Resize(jacobianShape);
  auto* drivingJacobianData = drivingJacobian->mutable_data<float>();

  auto drivingValue = generator_predictor_->GetInput(4);
  drivingValue->Resize(valueShape);
  auto* drivingValueData = drivingValue->mutable_data<float>();

  for (int i = 0; i < driving_num; i+=1) {
    t = GetCurrentTime();
    memcpy(drivingJacobianData, driving_jacobians[i], jacobianShapeNum*sizeof(float));
    memcpy(drivingValueData, driving_values[i], valueShapeNum*sizeof(float));
    t = GetCurrentTime();

    generator_predictor_->Run();
    LOGD("generator predict costs %d, %f ms", i, *predictGenTime);
    auto resultTensor = generator_predictor_->GetOutput(0);
    auto resultData = resultTensor->data<float>();


    cv::Mat res = cv::Mat::zeros(256, 256, CV_32FC3);
    int index = 0;
    for (int h = 0; h < 256; ++h) {
        for (int w = 0; w < 256; ++w) {
            //LOGD("pixel %d, %f, %f, %f", h*256+w, resultData[index], resultData[256*256+index], resultData[2*256*256+index]);
            res.at<cv::Vec3f>(h, w) = {resultData[index],
                                       resultData[256 * 256 + index],
                                       resultData[2 * 256 * 256 + index]}; // R,G,B

            index += 1; // update STEP times
        }
    }
    cv::Mat bgrImage;
    cv::cvtColor(res, bgrImage, cv::COLOR_RGB2BGR);
    cv::Mat temp = cv::Mat::zeros(bgrImage.size(), CV_8UC3);
    (bgrImage).convertTo(temp, CV_8UC3, 255);
    results.push_back(temp);

    *predictGenTime = GetElapsedTime(t);
  }


}

Fom::~Fom() {
    for (auto jacobian: driving_jacobians) {
      if (jacobian != nullptr) {
        delete []jacobian;
      }
    }
  for (auto value: driving_values) {
    if (value != nullptr) {
      delete []value;
    }
  }
}

Pipeline::Pipeline(const std::string &modelDir, const std::string &fommodelDir, const std::string &labelPath,
                   const int cpuThreadNum, const std::string &cpuPowerMode,
                   int inputWidth, int inputHeight,
                   const std::vector<float> &inputMean,
                   const std::vector<float> &inputStd, float scoreThreshold) {

  //detector_.reset(new Detector(modelDir, labelPath, cpuThreadNum, cpuPowerMode,
  //                             inputWidth, inputHeight, inputMean, inputStd,
  //                             scoreThreshold));


  fom_.reset(new Fom(fommodelDir, cpuThreadNum, cpuPowerMode));
}

void Pipeline::VisualizeResults(const std::vector<RESULT> &results,
                                cv::Mat *rgbaImage) {
  int w = rgbaImage->cols;
  int h = rgbaImage->rows;
  for (int i = 0; i < results.size(); i++) {
    RESULT object = results[i];
    cv::Rect boundingBox =
        cv::Rect(object.x * w, object.y * h, object.w * w, object.h * h) &
        cv::Rect(0, 0, w - 1, h - 1);
    // Configure text size
    std::string text = object.class_name + " ";
    text += std::to_string(static_cast<int>(object.score * 100)) + "%";
    int fontFace = cv::FONT_HERSHEY_PLAIN;
    double fontScale = 1.5f;
    float fontThickness = 1.0f;
    cv::Size textSize =
        cv::getTextSize(text, fontFace, fontScale, fontThickness, nullptr);
    // Draw roi object, text, and background
    cv::rectangle(*rgbaImage, boundingBox, object.fill_color, 2);
    cv::rectangle(*rgbaImage,
                  cv::Point2d(boundingBox.x,
                              boundingBox.y - round(textSize.height * 1.25f)),
                  cv::Point2d(boundingBox.x + boundingBox.width, boundingBox.y),
                  object.fill_color, -1);
    cv::putText(*rgbaImage, text, cv::Point2d(boundingBox.x, boundingBox.y),
                fontFace, fontScale, cv::Scalar(255, 255, 255), fontThickness);
  }
}

void Pipeline::VisualizeStatus(double readGLFBOTime, double writeGLTextureTime,
                               double preprocessTime, double predictTime,
                               double postprocessTime, cv::Mat *rgbaImage) {
  char text[255];
  cv::Scalar fontColor = cv::Scalar(255, 255, 255);
  int fontFace = cv::FONT_HERSHEY_PLAIN;
  double fontScale = 1.f;
  float fontThickness = 1;
  sprintf(text, "Read GLFBO time: %.1f ms", readGLFBOTime);
  cv::Size textSize =
      cv::getTextSize(text, fontFace, fontScale, fontThickness, nullptr);
  textSize.height *= 1.25f;
  cv::Point2d offset(10, textSize.height + 15);
  cv::putText(*rgbaImage, text, offset, fontFace, fontScale, fontColor,
              fontThickness);
  sprintf(text, "Write GLTexture time: %.1f ms", writeGLTextureTime);
  offset.y += textSize.height;
  cv::putText(*rgbaImage, text, offset, fontFace, fontScale, fontColor,
              fontThickness);
  sprintf(text, "Preprocess time: %.1f ms", preprocessTime);
  offset.y += textSize.height;
  cv::putText(*rgbaImage, text, offset, fontFace, fontScale, fontColor,
              fontThickness);
  sprintf(text, "Predict time: %.1f ms", predictTime);
  offset.y += textSize.height;
  cv::putText(*rgbaImage, text, offset, fontFace, fontScale, fontColor,
              fontThickness);
  sprintf(text, "Postprocess time: %.1f ms", postprocessTime);
  offset.y += textSize.height;
  cv::putText(*rgbaImage, text, offset, fontFace, fontScale, fontColor,
              fontThickness);
}

bool Pipeline::Process(int inTexureId, int outTextureId, int textureWidth,
                       int textureHeight, std::string savedImagePath) {
  static double readGLFBOTime = 0, writeGLTextureTime = 0;
  double preprocessTime = 0, predictTime = 0, postprocessTime = 0;

  // Read pixels from FBO texture to CV image
  cv::Mat rgbaImage;
  CreateRGBAImageFromGLFBOTexture(textureWidth, textureHeight, &rgbaImage,
                                  &readGLFBOTime);

  // Feed the image, run inference and parse the results
  std::vector<RESULT> results;
  detector_->Predict(rgbaImage, &results, &preprocessTime, &predictTime,
                     &postprocessTime);

  // Visualize the objects to the origin image
  VisualizeResults(results, &rgbaImage);

  // Visualize the status(performance data) to the origin image
  VisualizeStatus(readGLFBOTime, writeGLTextureTime, preprocessTime,
                  predictTime, postprocessTime, &rgbaImage);

  // Dump modified image if savedImagePath is set
  if (!savedImagePath.empty()) {
    cv::Mat bgrImage;
    cv::cvtColor(rgbaImage, bgrImage, cv::COLOR_RGBA2BGR);
    imwrite(savedImagePath, bgrImage);
  }

  // Write back to texture2D
  WriteRGBAImageBackToGLTexture(rgbaImage, outTextureId, &writeGLTextureTime);
  return true;
}

int Pipeline::FomProcess(std::string imagePath, std::string videoPath) {

  double predictGenTime = 0, predictKpTime = 0;
  cv::Mat source;
  source = cv::imread(imagePath);
  // Feed the image, run inference and parse the results
  std::vector<cv::Mat> results;
  fom_->Predict(source, videoPath, results, &predictKpTime, &predictGenTime);

  LOGD("finish predict");

  cv::VideoWriter writer("/storage/emulated/0/fom_inference001.avi",
                         cv::VideoWriter::fourcc('M', 'J', 'P', 'G'),
                         30, cv::Size(256, 256), true);

  for (int i = 0; i < results.size(); i++) {
    writer.write(results[i]);
  }
  writer.release();



  LOGD("finish write, %d", results.size());
  return true;
}
