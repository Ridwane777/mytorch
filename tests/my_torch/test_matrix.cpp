#include <criterion/criterion.h>

#include "my_torch/my_torch.hpp"

using my_torch::Matrix;

Test(matrix, is_filled_with_value)
{
    Matrix m(2, 3, 1.5);

    cr_assert_eq(m.rows(), 2);
    cr_assert_eq(m.cols(), 3);
    cr_assert_eq(m.size(), 6);
    cr_assert_eq(m(1, 2), 1.5);
}

Test(matrix, transpose_swaps_dimensions)
{
    Matrix m(2, 3);
    m(0, 2) = 7;

    Matrix t = m.transpose();
    cr_assert_eq(t.rows(), 3);
    cr_assert_eq(t.cols(), 2);
    cr_assert_eq(t(2, 0), 7);
}

Test(matrix, matmul_computes_product)
{
    Matrix a(2, 2);
    Matrix b(2, 1);
    a(0, 0) = 1;
    a(0, 1) = 2;
    a(1, 0) = 3;
    a(1, 1) = 4;
    b(0, 0) = 5;
    b(1, 0) = 6;

    Matrix c = a.matmul(b);
    cr_assert_eq(c(0, 0), 17);
    cr_assert_eq(c(1, 0), 39);
}

Test(matrix, matmul_rejects_wrong_dimensions)
{
    Matrix a(2, 3);
    Matrix b(2, 3);

    cr_assert_throw(a.matmul(b), my_torch::Error);
}
