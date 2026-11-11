// Prints how long a fixed mix of arithmetic and random memory accesses takes, in nanoseconds. On a
// machine in everyday use compare_builds.py runs it around the measurements: other work on the host,
// which a virtual machine cannot see in its own load figures, shows up here as a slower probe.
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <numeric>
#include <random>
#include <vector>

int main() {
    std::vector<uint32_t> next(size_t(1) << 22);  // 16 MB: beyond this laptop's L3
    std::iota(next.begin(), next.end(), 0);
    std::mt19937 generator(1);
    for (size_t index = next.size() - 1; index > 0; --index) {  // one cycle over all the elements
        std::swap(next[index], next[generator() % index]);
    }

    double best = 1e18;
    uint64_t sink = 0;
    for (int pass = 0; pass < 3; ++pass) {
        const auto started = std::chrono::steady_clock::now();
        uint32_t position = 0;
        uint64_t mix = 1;
        for (int step = 0; step < 1'000'000; ++step) {
            position = next[position];
            mix = mix * 0x9E3779B97F4A7C15ull + position;
        }
        sink += mix;
        const std::chrono::duration<double, std::nano> elapsed = std::chrono::steady_clock::now() - started;
        best = elapsed.count() < best ? elapsed.count() : best;
    }
    std::printf("%.0f %llu\n", best, static_cast<unsigned long long>(sink));
    return 0;
}
