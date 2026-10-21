#include <criterion/criterion.h>

#include "../../src/analyzer/Cli.hpp"

Test(cli, unimplemented_exits_with_84)
{
    cr_assert_eq(analyzer::run({}), analyzer::EXIT_ERROR);
}
