# Bonus (the subject requires bonus/ and allows a dedicated Makefile). The bonus reuses the FEN
# parser from batch B; the main code never depends on bonus/.
CXX		?=	g++
CXXFLAGS	+=	-std=c++20 -Wall -Wextra -Werror -I../include -I../src/analyzer -Ioracle -Ipositions
RELEASE		=	-O2

ORACLE		=	my_torch_oracle
ORACLE_SRC	=	oracle/Oracle.cpp ../src/analyzer/Fen.cpp

POSITIONS	=	my_torch_positions
POSITIONS_SRC	=	positions/Positions.cpp $(ORACLE_SRC)

EDA		=	my_torch_eda

TESTS		=	unit_tests
TESTS_SRC	=	$(wildcard tests/*.cpp)

all:	$(ORACLE) $(POSITIONS) $(EDA)

$(ORACLE):	oracle/main.cpp $(ORACLE_SRC) oracle/Oracle.hpp ../src/analyzer/Fen.hpp
	$(CXX) $(CXXFLAGS) $(RELEASE) -o $@ oracle/main.cpp $(ORACLE_SRC) $(LDFLAGS)

$(POSITIONS):	positions/main.cpp $(POSITIONS_SRC) positions/Positions.hpp oracle/Oracle.hpp
	$(CXX) $(CXXFLAGS) $(RELEASE) -o $@ positions/main.cpp $(POSITIONS_SRC) $(LDFLAGS)

$(EDA):	eda/main.cpp $(ORACLE_SRC) oracle/Oracle.hpp
	$(CXX) $(CXXFLAGS) $(RELEASE) -o $@ eda/main.cpp $(ORACLE_SRC) $(LDFLAGS)

tests_run:
	$(CXX) $(CXXFLAGS) -O2 -o $(TESTS) $(TESTS_SRC) $(POSITIONS_SRC) -lcriterion $(LDFLAGS)
	./$(TESTS)

clean:
	rm -f *.gcda *.gcno

fclean:	clean
	rm -f $(ORACLE) $(POSITIONS) $(EDA) $(TESTS)

re:	fclean all

.PHONY: all tests_run clean fclean re
