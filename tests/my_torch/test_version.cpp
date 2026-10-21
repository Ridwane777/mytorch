#include <criterion/criterion.h>

#include <string>

#include "my_torch/my_torch.hpp"

Test(my_torch, version_is_set)
{
    cr_assert(!std::string(my_torch::version()).empty());
}
