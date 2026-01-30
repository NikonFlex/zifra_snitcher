#include "stitcher.h"

#include <algorithm>
#include <cmath>

bool Stitcher::is_horizontal(const cv::Vec4i &line, double tan_max) {
  const double dx = static_cast<double>(line[2] - line[0]);
  const double dy = static_cast<double>(line[3] - line[1]);
  return std::abs(dy) <= std::abs(dx) * tan_max;
}

int Stitcher::seam_row_from_lines(const std::vector<cv::Vec4i> &lines, int width) {
  const double tan_max = std::tan(MAX_HORIZONTAL_ANGLE_RAD);
  double sum_len = 0.0;
  double sum_y = 0.0;
  for (const auto &line : lines) {
    if (!is_horizontal(line, tan_max)) {
      continue;
    }
    const double dx = static_cast<double>(line[2] - line[0]);
    const double dy = static_cast<double>(line[3] - line[1]);
    const double len = std::hypot(dx, dy);
    const double y = 0.5 * (line[1] + line[3]);
    sum_len += len;
    sum_y += len * y;
  }
  if (sum_len < width * MIN_LINE_SUPPORT_FRACTION) {
    return -1;
  }
  return static_cast<int>(sum_y / sum_len);
}

int Stitcher::seam_row_from_gradient(const cv::Mat &gray) {
  cv::Mat grad_y;
  cv::Sobel(gray, grad_y, CV_32F, 0, 1, 3);
  cv::Mat abs_y = cv::abs(grad_y);
  cv::Mat row_sum;
  cv::reduce(abs_y, row_sum, 1, cv::REDUCE_SUM, CV_64F);
  cv::Mat row_sum_smooth;
  cv::GaussianBlur(row_sum, row_sum_smooth, cv::Size(1, ROW_SMOOTH_KERNEL), 0);
  double max_val = 0.0;
  for (int y = 0; y < row_sum_smooth.rows; ++y) {
    max_val = std::max(max_val, row_sum_smooth.at<double>(y, 0));
  }
  if (max_val <= 0.0) {
    return -1;
  }
  const double mid = 0.5 * static_cast<double>(gray.rows);
  const double sigma = CENTER_SIGMA_FRACTION * static_cast<double>(gray.rows);
  double best_val = -1.0;
  int best_row = -1;
  for (int y = 0; y < row_sum_smooth.rows; ++y) {
    const double normalized = row_sum_smooth.at<double>(y, 0) / max_val;
    const double center_weight = std::exp(-0.5 * ((y - mid) * (y - mid)) / (sigma * sigma));
    const double score = normalized * center_weight;
    if (score > best_val) {
      best_val = score;
      best_row = y;
    }
  }
  return best_row;
}

double Stitcher::median(std::vector<double> &values) {
  if (values.empty()) {
    return 0.0;
  }
  const size_t mid = values.size() / 2;
  std::nth_element(values.begin(), values.begin() + mid, values.end());
  double med = values[mid];
  if (values.size() % 2 == 0) {
    std::nth_element(values.begin(), values.begin() + mid - 1, values.end());
    med = (med + values[mid - 1]) * 0.5;
  }
  return med;
}

cv::Mat Stitcher::stitch_sequence(const std::vector<cv::Mat> &images, const std::vector<int> &indices) const {
  if (images.empty()) {
    return {};
  }
  if (images.size() == 1) {
    return images.front().clone();
  }
  (void)indices;
  std::vector<cv::Point2i> offsets(images.size(), cv::Point2i(0, 0));
  for (size_t i = 1; i < images.size(); ++i) {
    const cv::Point2i shift = estimate_shift_split(images[i - 1], images[i]);
    offsets[i] = offsets[i - 1] + shift;
  }
  return stitch_images(images, offsets);
}

cv::Rect Stitcher::detect_seam_band(const cv::Mat &gray) const {
  cv::Mat blurred;
  cv::GaussianBlur(gray, blurred, cv::Size(5, 5), 1.5);
  cv::Scalar mean;
  cv::Scalar stddev;
  cv::meanStdDev(blurred, mean, stddev);
  const double lower = std::clamp(mean[0] - stddev[0], CANNY_LOWER_MIN, CANNY_LOWER_MAX);
  const double upper = std::clamp(mean[0] + CANNY_UPPER_STD_MUL * stddev[0], CANNY_UPPER_MIN, CANNY_UPPER_MAX);
  cv::Mat edges;
  cv::Canny(blurred, edges, lower, upper, CANNY_APERTURE, true);
  std::vector<cv::Vec4i> lines;
  const int min_len = std::max(HOUGH_MIN_LENGTH_PX, gray.cols / 4);
  cv::HoughLinesP(edges, lines, 1, CV_PI / 180.0, HOUGH_THRESHOLD, min_len, HOUGH_MAX_GAP);
  int seam_row = seam_row_from_lines(lines, gray.cols);
  if (seam_row < 0) {
    seam_row = seam_row_from_gradient(gray);
  }
  if (seam_row < 0) {
    seam_row = gray.rows / 2;
  }
  const int band_h = std::clamp(static_cast<int>(gray.rows * SEAM_BAND_FRACTION), SEAM_BAND_MIN_PX, gray.rows);
  const int y0 = std::clamp(seam_row - band_h / 2, 0, gray.rows - band_h);
  return cv::Rect(0, y0, gray.cols, band_h);
}

Stitcher::match_result Stitcher::match_shift(const cv::Mat &gray1, const cv::Mat &gray2, const cv::Mat &mask1, const cv::Mat &mask2) const {
  match_result result;
  cv::Ptr<cv::ORB> orb = cv::ORB::create(ORB_FEATURES);
  std::vector<cv::KeyPoint> kp1;
  std::vector<cv::KeyPoint> kp2;
  cv::Mat desc1;
  cv::Mat desc2;
  orb->detectAndCompute(gray1, mask1, kp1, desc1);
  orb->detectAndCompute(gray2, mask2, kp2, desc2);
  if (desc1.empty() || desc2.empty()) {
    return result;
  }
  cv::BFMatcher matcher(cv::NORM_HAMMING);
  std::vector<std::vector<cv::DMatch>> knn;
  matcher.knnMatch(desc1, desc2, knn, 2);
  std::vector<cv::DMatch> good;
  good.reserve(knn.size());
  for (const auto &m : knn) {
    if (m.size() < 2) {
      continue;
    }
    if (m[0].distance < MATCH_RATIO_TEST * m[1].distance) {
      good.push_back(m[0]);
    }
  }
  if (good.size() < static_cast<size_t>(MIN_MATCHES)) {
    return result;
  }
  std::vector<cv::Point2f> pts1;
  std::vector<cv::Point2f> pts2;
  pts1.reserve(good.size());
  pts2.reserve(good.size());
  for (const auto &m : good) {
    pts1.push_back(kp1[m.queryIdx].pt);
    pts2.push_back(kp2[m.trainIdx].pt);
  }
  cv::Mat inliers;
  const cv::Mat A = cv::estimateAffinePartial2D(pts2, pts1, inliers, cv::RANSAC, RANSAC_THRESHOLD, RANSAC_MAX_ITERS, RANSAC_CONFIDENCE, RANSAC_REFINE_ITERS);
  if (A.empty()) {
    return result;
  }
  std::vector<double> dxs;
  std::vector<double> dys;
  dxs.reserve(good.size());
  dys.reserve(good.size());
  int inlier_count = 0;
  for (int i = 0; i < inliers.rows; ++i) {
    if (inliers.at<uchar>(i, 0)) {
      dxs.push_back(pts1[i].x - pts2[i].x);
      dys.push_back(pts1[i].y - pts2[i].y);
      ++inlier_count;
    }
  }
  if (inlier_count == 0) {
    return result;
  }
  result.dx = cvRound(median(dxs));
  result.dy = cvRound(median(dys));
  result.ok = true;
  return result;
}

cv::Point2i Stitcher::estimate_shift_split(const cv::Mat &img1, const cv::Mat &img2) const {
  const int common_w = std::min(img1.cols, img2.cols);
  const int common_h = std::min(img1.rows, img2.rows);
  const cv::Rect common_roi(0, 0, common_w, common_h);
  cv::Mat gray1;
  cv::Mat gray2;
  cv::cvtColor(img1(common_roi), gray1, cv::COLOR_BGR2GRAY);
  cv::cvtColor(img2(common_roi), gray2, cv::COLOR_BGR2GRAY);
  const cv::Rect seam_band = detect_seam_band(gray1);
  cv::Mat mask_seam(gray1.size(), CV_8U, cv::Scalar(0));
  cv::Mat mask_bg(gray1.size(), CV_8U, cv::Scalar(255));
  mask_seam(seam_band).setTo(cv::Scalar(255));
  mask_bg(seam_band).setTo(cv::Scalar(0));
  const match_result seam_res = match_shift(gray1, gray2, mask_seam, mask_seam);
  const match_result bg_res = match_shift(gray1, gray2, mask_bg, mask_bg);
  cv::Mat gray1_f;
  cv::Mat gray2_f;
  gray1.convertTo(gray1_f, CV_32F);
  gray2.convertTo(gray2_f, CV_32F);
  cv::Mat hann;
  cv::createHanningWindow(hann, gray1_f.size(), CV_32F);
  const cv::Point2d phase_shift = cv::phaseCorrelate(gray1_f, gray2_f, hann);
  const int phase_dx = cvRound(phase_shift.x);
  const int phase_dy = cvRound(phase_shift.y);
  int dx = phase_dx;
  int dy = phase_dy;
  if (bg_res.ok) { dx = bg_res.dx; } else if (seam_res.ok) { dx = seam_res.dx; }
  if (seam_res.ok) { dy = seam_res.dy; } else if (bg_res.ok) { dy = bg_res.dy; }
  return cv::Point2i(dx, dy);
}

cv::Mat Stitcher::stitch_images(const std::vector<cv::Mat> &images, const std::vector<cv::Point2i> &offsets) const {
  int min_x = 0;
  int min_y = 0;
  int max_x = images[0].cols;
  int max_y = images[0].rows;
  for (size_t i = 0; i < images.size(); ++i) {
    const int x0 = offsets[i].x;
    const int y0 = offsets[i].y;
    const int x1 = x0 + images[i].cols;
    const int y1 = y0 + images[i].rows;
    min_x = std::min(min_x, x0);
    min_y = std::min(min_y, y0);
    max_x = std::max(max_x, x1);
    max_y = std::max(max_y, y1);
  }
  const int canvas_w = max_x - min_x;
  const int canvas_h = max_y - min_y;
  cv::Mat canvas(canvas_h, canvas_w, CV_8UC3, cv::Scalar(0, 0, 0));
  cv::Mat coverage(canvas_h, canvas_w, CV_8U, cv::Scalar(0));
  for (size_t i = 0; i < images.size(); ++i) {
    const cv::Point offset(offsets[i].x - min_x, offsets[i].y - min_y);
    const cv::Rect dst(offset.x, offset.y, images[i].cols, images[i].rows);
    if (i == 0) { images[i].copyTo(canvas(dst)); coverage(dst).setTo(cv::Scalar(255)); continue; }
    cv::Mat mask(images[i].rows, images[i].cols, CV_8U, cv::Scalar(255));
    cv::Mat covered = coverage(dst);
    cv::Mat col_max;
    cv::reduce(covered, col_max, 0, cv::REDUCE_MAX, CV_8U);
    int first = -1;
    int last = -1;
    for (int x = 0; x < col_max.cols; ++x) { if (col_max.at<uchar>(0, x) > 0) { if (first < 0) { first = x; } last = x; } }
    if (first >= 0) { const cv::Rect cut(first, 0, last - first + 1, images[i].rows); mask(cut).setTo(cv::Scalar(0)); }
    images[i].copyTo(canvas(dst), mask);
    if (first >= 0) {
      const int left_edge = dst.x + first;
      const int right_edge = dst.x + last + 1;
      const int strip_w = BLUR_STRIP_PX * 2;
      const int lx = std::max(dst.x, left_edge - BLUR_STRIP_PX);
      const int rx = std::max(dst.x, right_edge - BLUR_STRIP_PX);
      const int max_w = std::min(strip_w, canvas.cols - lx);
      if (max_w > 1) { cv::Mat strip = canvas(cv::Rect(lx, dst.y, max_w, dst.height)); cv::GaussianBlur(strip, strip, cv::Size(BLUR_KERNEL, BLUR_KERNEL), 0); }
      const int max_w2 = std::min(strip_w, canvas.cols - rx);
      if (max_w2 > 1) { cv::Mat strip2 = canvas(cv::Rect(rx, dst.y, max_w2, dst.height)); cv::GaussianBlur(strip2, strip2, cv::Size(BLUR_KERNEL, BLUR_KERNEL), 0); }
    }
    coverage(dst).setTo(cv::Scalar(255));
  }
  return canvas;
}
