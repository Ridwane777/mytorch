// Reproduces the measurement of ADR-0007: the gradient of the cross-entropy through a Softmax,
// computed the naive way (dL/dp = -t / max(p, eps), then Softmax::backward) and the coupled way
// (CrossEntropy::gradient), against the exact value p - t.
//
// Build and run from the repository root:
//   g++ -std=c++20 -Iinclude scripts/softmax_ce_gradient.cpp src/my_torch/*.cpp -o /tmp/ce &&
//   /tmp/ce
#include <algorithm>
#include <cstdio>

#include "my_torch/my_torch.hpp"

int main()
{
    using namespace my_torch;
    constexpr double epsilon = 1e-15;

    std::printf("gap  p(right class)  naive      coupled    exact (p - t)\n");
    for (double gap : {10.0, 30.0, 40.0}) {
        Matrix z(1, 2);
        Matrix t(1, 2);
        Softmax softmax;

        z(0, 0) = -gap; // the right class (index 0) is pushed down
        t(0, 0) = 1;
        const Matrix p = softmax.forward(z, true);
        Matrix naive(1, 2);
        naive(0, 0) = -1.0 / std::max(p(0, 0), epsilon);
        const double naiveGradient = softmax.backward(naive)(0, 0);
        const double coupled = CrossEntropy().gradient(p, t)(0, 0);

        std::printf("%3.0f  %.3e       %+.6f  %+.6f  %+.6f\n", gap, p(0, 0), naiveGradient, coupled,
                    p(0, 0) - 1);
    }
    return 0;
}
