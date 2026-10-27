CXX		?=	g++
CXXFLAGS	+=	-std=c++20 -Wall -Wextra -Werror -Iinclude -MMD -MP -pthread
LDFLAGS		+=	-pthread
RELEASE		=	-O2
BUILD		=	build

LIB		=	$(BUILD)/libmy_torch.a
LIB_SRC		=	$(wildcard src/my_torch/*.cpp)
LIB_OBJ		=	$(LIB_SRC:%.cpp=$(BUILD)/%.o)

ANALYZER	=	my_torch_analyzer
ANALYZER_SRC	=	$(filter-out src/analyzer/main.cpp, $(wildcard src/analyzer/*.cpp))
ANALYZER_OBJ	=	$(ANALYZER_SRC:%.cpp=$(BUILD)/%.o)

GENERATOR	=	my_torch_generator
GENERATOR_SRC	=	$(filter-out src/generator/main.cpp, $(wildcard src/generator/*.cpp))
GENERATOR_OBJ	=	$(GENERATOR_SRC:%.cpp=$(BUILD)/%.o)

TESTS		=	unit_tests
TESTS_SRC	=	$(wildcard tests/*/*.cpp)

all:	$(ANALYZER) $(GENERATOR)

$(LIB):	$(LIB_OBJ)
	ar rcs $@ $^

$(ANALYZER):	$(BUILD)/src/analyzer/main.o $(ANALYZER_OBJ) $(LIB)
	$(CXX) -o $@ $^ $(LDFLAGS)

$(GENERATOR):	$(BUILD)/src/generator/main.o $(GENERATOR_OBJ) $(LIB)
	$(CXX) -o $@ $^ $(LDFLAGS)

$(BUILD)/%.o:	%.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $(RELEASE) -c $< -o $@

debug:	RELEASE = -O0 -g3 -fsanitize=address,undefined
debug:	fclean
	$(MAKE) RELEASE="$(RELEASE)" LDFLAGS="-fsanitize=address,undefined"

tests_run:
	$(CXX) $(CXXFLAGS) -O0 --coverage -o $(TESTS) $(TESTS_SRC) $(LIB_SRC) $(ANALYZER_SRC) \
		$(GENERATOR_SRC) -lcriterion
	./$(TESTS)

clean:
	rm -rf $(BUILD) *.gcda *.gcno $(TESTS).d

fclean:	clean
	rm -f $(ANALYZER) $(GENERATOR) $(TESTS)

re:	fclean all

-include $(shell find $(BUILD) -name '*.d' 2>/dev/null)

.PHONY: all debug tests_run clean fclean re
