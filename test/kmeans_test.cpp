//
// Created by Hervé Paulino on 02/04/2025.
//

#include <gtest/gtest.h>
#include <kmeans.h>
#include <fstream>

/**
 * Compares the contents of two files
 *
 * @param result_filename
 * @param expected_result_filename
 */
void compare_results(const std::string& result_filename, const std::string& expected_result_filename) {
    std::string bufferResult;
    std::string bufferExpected;
    std::ifstream result(result_filename);
    std::ifstream expected(expected_result_filename);

    ASSERT_TRUE(result.is_open());
    ASSERT_TRUE(expected.is_open());

    while (getline (result, bufferResult)) {
        getline (expected, bufferExpected);

        // Remove carriage return if present (Windows)
        if (!bufferExpected.empty() && bufferExpected.back() == '\r') {
            bufferExpected.pop_back();
        }

        EXPECT_STREQ(bufferResult.c_str(), bufferExpected.c_str());
    }

    result.close();
    expected.close();
}


TEST(KMeans, Input2D_10_1000_100_4) {
    const std::string output_filename = "output2D_100_100_100_4.txt";
    kmeans("../test_files/input2D.inp", 100, 100, 100, 0.4, output_filename.c_str());
    compare_results(output_filename, "../test/expected2D_100_100_100_4.txt");
}


TEST(KMeans, Input100D2_100_10_0_0005) {
    const std::string output_filename = "output100D2_100_10_0_0005.txt";
    kmeans("../test_files/input100D2.inp", 100, 10, 0, 0.005, output_filename.c_str());
    compare_results(output_filename, "../test/expected100D2_100_10_0_0005.txt");
}