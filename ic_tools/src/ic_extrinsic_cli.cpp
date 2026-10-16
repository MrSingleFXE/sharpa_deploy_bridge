// Headless industrial_calibration extrinsic hand-eye calibration.
// Same steps as industrial_calibration/examples/extrinsic_hand_eye.cpp (no image windows):
//   load image/pose pairs -> target finder plugin -> homography check -> optimize() -> analysis -> result YAML
// Usage: ic_extrinsic_cli <config.yaml> <cal_data.yaml> <result.yaml>
//   config.yaml   : intrinsics, camera_mount_to_camera_guess, target_mount_to_target_guess, target_finder,
//                   homography_threshold, static_camera   (the calibration app's configuration format)
//   cal_data.yaml : "data: [{image: ..., pose: ...}, ...]" written by the industrial_calibration_ros data collector
//   result.yaml   : the ExtrinsicHandEyeResult YAML (same as the app's "Save calibration")
#include <industrial_calibration/optimizations/extrinsic_hand_eye.h>
#include <industrial_calibration/analysis/homography_analysis.h>
#include <industrial_calibration/analysis/extrinsic_hand_eye_calibration_analysis.h>
#include <industrial_calibration/target_finders/opencv/target_finder.h>
#include <industrial_calibration/core/serialization.h>
#include <boost_plugin_loader/plugin_loader.hpp>
#include <opencv2/imgcodecs.hpp>
#include <filesystem>
#include <fstream>
#include <iostream>

using namespace industrial_calibration;
namespace fs = std::filesystem;

int main(int argc, char** argv)
{
  if (argc != 4)
  {
    std::cerr << "usage: ic_extrinsic_cli <config.yaml> <cal_data.yaml> <result.yaml>\n";
    return 2;
  }
  try
  {
    YAML::Node config = YAML::LoadFile(argv[1]);
    const fs::path data_file(argv[2]);
    YAML::Node data = YAML::LoadFile(data_file.string())["data"];

    ExtrinsicHandEyeProblem2D3D problem;
    const bool static_camera = getMember<bool>(config, "static_camera");
    const auto homography_threshold = getMember<double>(config, "homography_threshold");
    problem.target_mount_to_target_guess = getMember<Eigen::Isometry3d>(config, "target_mount_to_target_guess");
    problem.camera_mount_to_camera_guess = getMember<Eigen::Isometry3d>(config, "camera_mount_to_camera_guess");
    problem.intr = getMember<CameraIntrinsics>(config, "intrinsics");

    boost_plugin_loader::PluginLoader loader;
    loader.search_libraries.insert(INDUSTRIAL_CALIBRATION_PLUGIN_LIBRARIES);
    loader.search_libraries_env = INDUSTRIAL_CALIBRATION_SEARCH_LIBRARIES_ENV;
    YAML::Node tf_config = getMember<YAML::Node>(config, "target_finder");
    auto factory = loader.createInstance<TargetFinderFactoryOpenCV>(getMember<std::string>(tf_config, "type"));
    TargetFinderOpenCV::ConstPtr target_finder = factory->create(tf_config);

    std::size_t used = 0;
    for (std::size_t i = 0; i < data.size(); ++i)
    {
      const fs::path img_file = data_file.parent_path() / data[i]["image"].as<std::string>();
      const fs::path pose_file = data_file.parent_path() / data[i]["pose"].as<std::string>();
      try
      {
        cv::Mat image = cv::imread(img_file.string());
        if (image.empty())
          throw std::runtime_error("cannot read image");
        const auto pose = YAML::LoadFile(pose_file.string()).as<Eigen::Isometry3d>();
        Observation2D3D obs;
        obs.to_camera_mount = static_camera ? Eigen::Isometry3d::Identity() : pose;
        obs.to_target_mount = static_camera ? pose : Eigen::Isometry3d::Identity();
        obs.correspondence_set = target_finder->findCorrespondences(image);
        RandomCorrespondenceSampler sampler(obs.correspondence_set.size(), obs.correspondence_set.size() / 2, 1);
        Eigen::VectorXd h = calculateHomographyError(obs.correspondence_set, sampler);
        if (h.array().mean() > homography_threshold)
          throw std::runtime_error("homography error " + std::to_string(h.array().mean()) + " > threshold");
        problem.observations.push_back(obs);
        ++used;
        std::cout << "observation " << i << ": " << obs.correspondence_set.size() << " correspondences\n";
      }
      catch (const std::exception& ex)
      {
        std::cout << "observation " << i << " skipped: " << ex.what() << "\n";
      }
    }
    std::cout << "using " << used << " / " << data.size() << " observations\n";
    if (used < 3)
      throw std::runtime_error("not enough valid observations");

    ExtrinsicHandEyeResult result = optimize(problem);
    std::cout << "\n" << result << "\n";
    std::cout << result.covariance.printCorrelationCoeffAboveThreshold(0.5) << "\n";
    std::cout << analyze3dProjectionError(problem, result) << "\n\n";
    std::cout << analyzeResults(problem, result) << "\n";

    std::ofstream f(argv[3]);
    f << YAML::Node(result);
    std::cout << "\nsaved " << argv[3] << "\n";
    return result.converged ? 0 : 1;
  }
  catch (const std::exception& ex)
  {
    std::cerr << "ERROR: " << ex.what() << "\n";
    return 1;
  }
}
