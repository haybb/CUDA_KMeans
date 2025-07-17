#include <kmeans.h>

int main() {
    kmeans("../test_files/input100D2.inp", 100, 10, 0, 0.005, "../test/expected100D2_100_10_0_0005.txt");
    return 0;
}