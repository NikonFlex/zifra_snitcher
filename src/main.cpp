#include <algorithm>
#include <filesystem>
#include <iostream>
#include <string>
#include <unordered_set>
#include <vector>

#include <opencv2/opencv.hpp>

#include "stitcher.h"

struct args { std::vector<int> exclude; };

void parse_exclude_list(const std::string &value, std::vector<int> &out) {
  size_t start = 0;
  while (start < value.size()) {
    size_t end = value.find(',', start);
    const std::string token = value.substr(start, end == std::string::npos ? value.size() - start : end - start);
    if (!token.empty()) { try { out.push_back(std::stoi(token)); } catch (...) {} }
    if (end == std::string::npos) { break; }
    start = end + 1;
  }
}

args parse_args(int argc, char **argv) {
  args parsed;
  for (int i = 1; i < argc; ++i) {
    const std::string token = argv[i];
    if (token == "--exclude" && i + 1 < argc) { parse_exclude_list(argv[++i], parsed.exclude); }
  }
  return parsed;
}

std::vector<std::pair<int, std::filesystem::path>> list_images(const std::string &input_dir, const std::vector<int> &exclude) {
  std::unordered_set<int> exclude_set(exclude.begin(), exclude.end());
  std::vector<std::pair<int, std::filesystem::path>> files;
  for (const auto &entry : std::filesystem::directory_iterator(input_dir)) {
    if (!entry.is_regular_file()) { continue; }
    const auto &path = entry.path();
    const std::string ext = path.extension().string();
    if (ext != ".jpg" && ext != ".jpeg" && ext != ".png") { continue; }
    const std::string stem = path.stem().string();
    try { int idx = std::stoi(stem); if (exclude_set.count(idx)) { continue; } files.emplace_back(idx, path); } catch (...) { continue; }
  }
  std::sort(files.begin(), files.end(), [](const auto &a, const auto &b) { return a.first < b.first; });
  std::vector<std::pair<int, std::filesystem::path>> ordered;
  ordered.reserve(files.size());
  const int priority[] = {0, 1, 2, 3};
  for (const auto &item : files) { if (item.first == 0 || item.first == 1 || item.first == 2 || item.first == 3) { continue; } ordered.push_back(item); }
  for (int idx : priority) { auto it = std::find_if(files.begin(), files.end(), [idx](const auto &item) { return item.first == idx; }); if (it != files.end()) { ordered.push_back(*it); } }
  return ordered;
}

int main(int argc, char **argv) {
  const args parsed = parse_args(argc, argv);
  const std::string input_dir = "input";
  const auto files = list_images(input_dir, parsed.exclude);
  if (files.size() < 2) { std::cerr << "Need at least two numbered images in " << input_dir << std::endl; return 1; }
  std::vector<cv::Mat> images;
  std::vector<int> indices;
  images.reserve(files.size());
  indices.reserve(files.size());
  for (const auto &f : files) {
    cv::Mat img = cv::imread(f.second.string(), cv::IMREAD_COLOR);
    if (img.empty()) { std::cerr << "Failed to read image: " << f.second << std::endl; return 1; }
    images.push_back(img);
    indices.push_back(f.first);
  }
  Stitcher stitcher;
  const cv::Mat canvas = stitcher.stitch_sequence(images, indices);
  const std::string out_path = "output/stitched_all.jpg";
  if (!cv::imwrite(out_path, canvas)) { std::cerr << "Failed to write output image: " << out_path << std::endl; return 1; }
  std::cout << "Saved stitched image to " << out_path << std::endl;
  return 0;
}
