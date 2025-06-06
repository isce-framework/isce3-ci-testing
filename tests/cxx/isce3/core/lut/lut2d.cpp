#include <iostream>
#include <cmath>
#include <vector>
#include <valarray>
#include <string>
#include <fstream>
#include <sstream>
#include "isce3/core/Matrix.h"
#include "isce3/core/LUT2d.h"
#include "isce3/core/Utilities.h"
#include "gtest/gtest.h"

void loadInterpData(isce3::core::Matrix<double> & M);


// Test biquintic LUT evaluation
TEST(LUT2dTest, Evaluation) {

    // Create indices
    std::vector<double> xvec = isce3::core::arange(-5.01, 5.01, 0.25);
    std::vector<double> yvec = isce3::core::arange(-5.01, 5.01, 0.25);
    size_t nx = xvec.size();
    size_t ny = yvec.size();

    // Copy them to valarrays
    std::valarray<double> xindex(xvec.data(), xvec.size());
    std::valarray<double> yindex(yvec.data(), yvec.size());

    // Allocate the matrix
    isce3::core::Matrix<double> M;
    M.resize(ny, nx);

    // Read the reference data
    isce3::core::Matrix<double> ref_values;
    loadInterpData(ref_values);
    const size_t N_pts = ref_values.length();

    // Fill matrix values with function z = sin(x**2 + y**2)
    for (size_t i = 0; i < ny; ++i) {
        for (size_t j = 0; j < nx; ++j) {
            M(i,j) = std::sin(yindex[i]*yindex[i] + xindex[j]*xindex[j]);
        }
    }

    // Create LUT2d
    isce3::core::LUT2d<double> lut(xindex, yindex, M, isce3::core::BIQUINTIC_METHOD);

    // Loop over test points
    double error = 0.0;
    for (size_t i = 0; i < N_pts; ++i) {
        // Perform evaluation
        const double z = lut.eval(ref_values(i,0), ref_values(i,1));
        // Accumulate error
        error += std::pow(z - ref_values(i,5), 2);
    }
    ASSERT_TRUE((error / N_pts) < 0.058);
}

void loadInterpData(isce3::core::Matrix<double> & M) {
    /*
    Load ground truth interpolation data. The test data is the function:

    z = sqrt(x^2 + y^2)

    The columns of the data are:
    x_index    y_index    bilinear_interp    bicubic_interp    5thorder_spline    truth
    */

    // Open file for reading
    std::ifstream fid(TESTDATA_DIR "interpolator/data.txt");
    // Check if file open was successful
    if (fid.fail()) {
        std::cout << "Error: Failed to open data file for interpolator test." << std::endl;
    }

    std::vector<double> xvec, yvec, zlinear_vec, zcubic_vec, zquintic_vec, ztrue_vec;

    // Loop over interpolation data
    while (fid) {

        // Parse line
        std::string str;
        std::stringstream stream;
        double x, y, z_linear, z_cubic, z_quintic, z_true;

        std::getline(fid, str);
        if (str.length() < 1)
            break;
        stream << str;
        stream >> x >> y >> z_linear >> z_cubic >> z_quintic >> z_true;

        // Add data to orbit
        xvec.push_back(x);
        yvec.push_back(y);
        zlinear_vec.push_back(z_linear);
        zcubic_vec.push_back(z_cubic);
        zquintic_vec.push_back(z_quintic);
        ztrue_vec.push_back(z_true);
    }

    // Close the file
    fid.close();

    // Fill the matrix
    const size_t N = xvec.size();
    M.resize(N, 6);
    for (size_t i = 0; i < N; ++i) {
        M(i,0) = xvec[i];
        M(i,1) = yvec[i];
        M(i,2) = zlinear_vec[i];
        M(i,3) = zcubic_vec[i];
        M(i,4) = zquintic_vec[i];
        M(i,5) = ztrue_vec[i];
    }
}

TEST(LUT2dTest, Contains)
{
    using T = float;

    {
        const double x0 = 0.;
        const double dx = 0.5;
        const size_t width = 3;

        const double y0 = 10.;
        const double dy = 0.25;
        const size_t length = 5;

        const auto data = isce3::core::Matrix<T>(length, width);
        const auto lut2d = isce3::core::LUT2d<T>(x0, y0, dx, dy, data);

        EXPECT_TRUE(lut2d.contains(lut2d.yStart(), lut2d.xStart()));
        EXPECT_TRUE(lut2d.contains(lut2d.yEnd(), lut2d.xEnd()));
        EXPECT_TRUE(lut2d.contains(10.5, 0.5));

        EXPECT_FALSE(lut2d.contains(9.9, 0.5));
        EXPECT_FALSE(lut2d.contains(11.1, 0.5));
        EXPECT_FALSE(lut2d.contains(10.5, -0.1));
        EXPECT_FALSE(lut2d.contains(10.5, 1.1));
    }

    {
        // Default-constructed LUT has infinite extent.
        const isce3::core::LUT2d<T> lut2d;
        const double bigval = 1e9;
        EXPECT_TRUE(lut2d.contains(bigval, bigval));
        EXPECT_TRUE(lut2d.contains(-bigval, -bigval));
    }
}

TEST(LUT2dTest, RefValue)
{
    using T = double;
    {
        // default ctor sets to zero
        auto lut = isce3::core::LUT2d<T>();
        EXPECT_EQ(lut.refValue(), 0.0);

        // setter should update value
        lut.refValue(1.0);
        EXPECT_EQ(lut.refValue(), 1.0);
    }
    {
        // scalar parameter ctor sets value
        auto lut = isce3::core::LUT2d<T>(2.0);
        EXPECT_EQ(lut.refValue(), 2.0);
    }
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
