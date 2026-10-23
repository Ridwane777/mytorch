# Bonus (le sujet impose bonus/ et autorise un Makefile dédié). Le bonus réutilise le parseur FEN
# du lot B ; le code principal ne dépend jamais de bonus/.
CXX		?=	g++
CXXFLAGS	+=	-std=c++20 -Wall -Wextra -Werror -I../include -I../src/analyzer -Ioracle
RELEASE		=	-O2

ORACLE		=	my_torch_oracle
ORACLE_SRC	=	oracle/Oracle.cpp ../src/analyzer/Fen.cpp

TESTS		=	unit_tests
TESTS_SRC	=	$(wildcard tests/*.cpp)

all:	$(ORACLE)

$(ORACLE):	oracle/main.cpp $(ORACLE_SRC) oracle/Oracle.hpp ../src/analyzer/Fen.hpp
	$(CXX) $(CXXFLAGS) $(RELEASE) -o $@ oracle/main.cpp $(ORACLE_SRC) $(LDFLAGS)

tests_run:
	$(CXX) $(CXXFLAGS) -O2 -o $(TESTS) $(TESTS_SRC) $(ORACLE_SRC) -lcriterion $(LDFLAGS)
	./$(TESTS)

clean:
	rm -f *.gcda *.gcno

fclean:	clean
	rm -f $(ORACLE) $(TESTS)

re:	fclean all

.PHONY: all tests_run clean fclean re
