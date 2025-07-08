// ceres_solver.cpp

#include "neo_localization/ceres_solver.h"
#include <iostream>

using ceres::AutoDiffCostFunction;
using ceres::CostFunction;
using ceres::Jet;

namespace
{

  // 泛型版本：处理 double、float 等普通标量类型
  template <typename U>
  inline float getScalar(const U &x)
  {
    return static_cast<float>(x);
  }

  // Jet 特化版本：用于 Ceres 自动微分
  // Jet<T, N>	自动微分用的类型，包含函数值和 N 个导数分量
  template <typename T, int N>
  inline float getScalar(const ceres::Jet<T, N> &x)
  {
    return static_cast<float>(x.a); // 只提取实数部分
  }

}

namespace
{
  // 基础残差（单层）
  // 在优化中，根据当前的位姿 pose，把激光点 p 转到地图上，对应插值得到的 地图值（map_val）作为残差 residual。
  struct ScanResidual
  {
    scan_point_t p;                             // 单个激光点
    std::shared_ptr<const GridMap<float>> grid; // 栅格地图

    ScanResidual(scan_point_t point, std::shared_ptr<const GridMap<float>> g)
        : p(point), grid(std::move(g)) {}
    template <typename T>
    bool operator()(const T *const pose, T *residual) const
    {
      T cos_yaw = ceres::cos(pose[2]);
      T sin_yaw = ceres::sin(pose[2]);

      T wx = pose[0] + cos_yaw * T(p.x) - sin_yaw * T(p.y);
      T wy = pose[1] + sin_yaw * T(p.x) + cos_yaw * T(p.y);

      float gx = grid->world_to_grid(getScalar(wx));
      float gy = grid->world_to_grid(getScalar(wy));
      float map_val = grid->bilinear_lookup(gx, gy);

      residual[0] = T(-map_val);
      return true;
    }
  };

  // 多层地图残差
  struct MultiLayerScanResidual
  {
    scan_point_ex_t p;
    std::shared_ptr<const MultiGridMap<float>> multi_grid;

    MultiLayerScanResidual(scan_point_ex_t point, std::shared_ptr<const MultiGridMap<float>> g)
        : p(point), multi_grid(std::move(g)) {}

    template <typename T>
    bool operator()(const T *const pose, T *residual) const
    {
      const auto &grid = multi_grid->layers[p.layer];

      T cos_yaw = ceres::cos(pose[2]);
      T sin_yaw = ceres::sin(pose[2]);

      T wx = pose[0] + cos_yaw * T(p.x) - sin_yaw * T(p.y);
      T wy = pose[1] + sin_yaw * T(p.x) + cos_yaw * T(p.y);

      float gx = grid->world_to_grid(getScalar(wx));
      float gy = grid->world_to_grid(getScalar(wy));
      float map_val = grid->bilinear_lookup(gx, gy);

      residual[0] = T(map_val) * T(p.w);
      return true;
    }
  };

} // namespace

// 构造函数
CeresScanMatcher::CeresScanMatcher(double gain, double damping)
  : gain_(gain), damping_(damping) {}

// 单层地图求解
void CeresScanMatcher::solve(
    const std::shared_ptr<const GridMap<float>>& grid,
    const std::vector<scan_point_t>& points,
    double& pose_x, double& pose_y, double& pose_yaw)
{
  double pose[3] = { pose_x, pose_y, pose_yaw };
  ceres::Problem problem;

  for (const auto& pt : points) {
    CostFunction* cost_function =
      new AutoDiffCostFunction<ScanResidual, 1, 3>(new ScanResidual(pt, grid));
    problem.AddResidualBlock(cost_function, nullptr, pose);
  }

  ceres::Solver::Options options;
  options.max_num_iterations = 20;
  options.linear_solver_type = ceres::DENSE_QR;
  options.minimizer_progress_to_stdout = false;

  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);
  //gain 控制步长：避免一次跳动过大
  pose_x += gain_ * (pose[0] - pose_x);
  pose_y += gain_ * (pose[1] - pose_y);
  pose_yaw += gain_ * (pose[2] - pose_yaw);
}

// 多层地图求解
// 多图层地图的求解过程与单层地图类似，但需要处理每个点的图层信息。
// 比如 反光板、二维码等不同图层的激光点，可能需要不同的权重（w）和图层索引（layer）。
void CeresScanMatcher::solve(
    const std::shared_ptr<const MultiGridMap<float>>& multi_grid,
    const std::vector<scan_point_ex_t>& points,
    double& pose_x, double& pose_y, double& pose_yaw)
{
  double pose[3] = { pose_x, pose_y, pose_yaw };
  ceres::Problem problem;

  for (const auto& pt : points) {
    CostFunction* cost_function =
      new AutoDiffCostFunction<MultiLayerScanResidual, 1, 3>(new MultiLayerScanResidual(pt, multi_grid));
    problem.AddResidualBlock(cost_function, nullptr, pose);
  }

  ceres::Solver::Options options;
  options.max_num_iterations = 20;
  options.linear_solver_type = ceres::DENSE_QR;
  options.minimizer_progress_to_stdout = false;

  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);
  r_norm = std::sqrt(summary.final_cost / points.size());// 计算均方根误差
  pose_x += gain_ * (pose[0] - pose_x);
  pose_y += gain_ * (pose[1] - pose_y);
  pose_yaw += gain_ * (pose[2] - pose_yaw);
}
