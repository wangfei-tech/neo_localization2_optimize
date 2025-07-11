// ceres_solver.h

#ifndef NEO_LOCALIZATION_CERES_SOLVER_H
#define NEO_LOCALIZATION_CERES_SOLVER_H

#include <memory>
#include <vector>
#include <ceres/ceres.h>
#include <neo_localization/GridMap.h>
#include <neo_localization/Util.h>
#include <ceres/jet.h>

class CeresScanMatcher {
public:
  CeresScanMatcher(double gain = 1.0, double damping = 1.0);

  void solve(const std::shared_ptr<const GridMap<float>>& grid,
             const std::vector<scan_point_t>& points,
             double& pose_x, double& pose_y, double& pose_yaw);

  void solve(const std::shared_ptr<const MultiGridMap<float>>& multi_grid,
             const std::vector<scan_point_ex_t>& points,
             double& pose_x, double& pose_y, double& pose_yaw);
  double gain_;
  double damping_;
  double r_norm = 0.0;          // current error norm
};

#endif  // NEO_LOCALIZATION_CERES_SOLVER_H
