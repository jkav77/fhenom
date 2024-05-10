#include <vector>

namespace fhenom {
static const std::vector<double> kF1Coeffs{0, 3.0 / 2, 0, -1.0 / 2};
static const std::vector<double> kF2Coeffs{0, 15.0 / 8, 0, -10.0 / 8, 0, 3.0 / 8};
static const std::vector<double> kF3Coeffs{0.0, 2.1875, 0.0, -2.1875, 0.0, 1.3125, 0.0, -0.3125};
static const std::vector<double> kF4Coeffs{0, 315.0 / 128,  0, -420.0 / 128, 0, 378.0 / 128,
                                           0, -180.0 / 128, 0, 35.0 / 128};
static const std::vector<double> kF7Coeffs{0, 3.14208984,  0, -7.33154297, 0, 13.19677734, 0, -15.71044922,
                                           0, 12.21923828, 0, -5.99853516, 0, 1.69189453,  0, -0.20947266};

// static const std::vector<double> kG3Coeffs{0.0, 4.477073170731708, 0.0,
// -16.1884765625,
//    0.0, 25.013671875,      0.0, -12.55859375};

static const std::vector<double> kG1Coeffs{0, 2.076171875, 0, -1.3271484375};
static const std::vector<double> kG2Coeffs{0, 3.255859375, 0, -5.96484375, 0, 3.70703125};
static const std::vector<double> kG3Coeffs{0, 4.4814453125, 0, -16.1884765625, 0, 25.013671875, 0, -12.55859375};
static const std::vector<double> kG4Coeffs{0, 2.076171875, 0, -1.3271484375};

}  // namespace fhenom