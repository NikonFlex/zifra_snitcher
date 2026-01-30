#ifndef ZIFRA_STITCHER_H
#define ZIFRA_STITCHER_H

#include <cstddef>
#include <vector>

#include <opencv2/opencv.hpp>

class Stitcher {
 public:
  Stitcher() = default;
  cv::Mat stitch_sequence(const std::vector<cv::Mat> &images, const std::vector<int> &indices) const;

 private:
  struct match_result {
    bool ok = false;
    int dx = 0;
    int dy = 0;
  };

  static constexpr int ORB_FEATURES = 2000;
  static constexpr float MATCH_RATIO_TEST = 0.75f;
  static constexpr int MIN_MATCHES = 8;
  static constexpr double RANSAC_THRESHOLD = 3.0;
  static constexpr int RANSAC_MAX_ITERS = 2000;
  static constexpr double RANSAC_CONFIDENCE = 0.99;
  static constexpr int RANSAC_REFINE_ITERS = 10;
  static constexpr double SEAM_BAND_FRACTION = 0.18;
  static constexpr int SEAM_BAND_MIN_PX = 30;
  static constexpr double MAX_HORIZONTAL_ANGLE_RAD = 10.0 * CV_PI / 180.0;
  static constexpr double MIN_LINE_SUPPORT_FRACTION = 0.30;
  static constexpr int CANNY_APERTURE = 3;
  static constexpr double CANNY_LOWER_MIN = 5.0;
  static constexpr double CANNY_LOWER_MAX = 150.0;
  static constexpr double CANNY_UPPER_MIN = 30.0;
  static constexpr double CANNY_UPPER_MAX = 255.0;
  static constexpr double CANNY_UPPER_STD_MUL = 2.0;
  static constexpr int HOUGH_THRESHOLD = 60;
  static constexpr int HOUGH_MIN_LENGTH_PX = 30;
  static constexpr int HOUGH_MAX_GAP = 20;
  static constexpr int ROW_SMOOTH_KERNEL = 31;
  static constexpr double CENTER_SIGMA_FRACTION = 0.30;
  static constexpr int BLUR_STRIP_PX = 10;
  static constexpr int BLUR_KERNEL = 7;

  static bool is_horizontal(const cv::Vec4i &line, double tan_max);
  static int seam_row_from_lines(const std::vector<cv::Vec4i> &lines, int width);
  static int seam_row_from_gradient(const cv::Mat &gray);
  static double median(std::vector<double> &values);

  cv::Rect detect_seam_band(const cv::Mat &gray) const;
  match_result match_shift(const cv::Mat &gray1, const cv::Mat &gray2, const cv::Mat &mask1, const cv::Mat &mask2) const;
  cv::Point2i estimate_shift_split(const cv::Mat &img1, const cv::Mat &img2) const;
  cv::Mat stitch_images(const std::vector<cv::Mat> &images, const std::vector<cv::Point2i> &offsets) const;
};

#endif
