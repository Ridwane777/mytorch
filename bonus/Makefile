# Bonus (the subject requires bonus/ and allows a dedicated Makefile). The bonus reuses the FEN
# parser from batch B; the main code never depends on bonus/.
CXX		?=	g++
CXXFLAGS	+=	-std=c++20 -Wall -Wextra -Werror -I../include -I../src/analyzer -Ioracle -Ipositions -Igui
RELEASE		=	-O2

ORACLE		=	my_torch_oracle
ORACLE_SRC	=	oracle/Oracle.cpp ../src/analyzer/Fen.cpp

POSITIONS	=	my_torch_positions
POSITIONS_SRC	=	positions/Positions.cpp $(ORACLE_SRC)

EDA		=	my_torch_eda

# Chess GUI (ADR-0029): links the library and the analyzer's encodings.
GUI		=	my_torch_gui
LIB		=	../build/libmy_torch.a
GUI_SRC		=	gui/Analysis.cpp ../src/analyzer/Encoding.cpp $(ORACLE_SRC)

TESTS		=	unit_tests
TESTS_SRC	=	$(wildcard tests/*.cpp)

all:	$(ORACLE) $(POSITIONS) $(EDA) $(GUI)

$(ORACLE):	oracle/main.cpp $(ORACLE_SRC) oracle/Oracle.hpp ../src/analyzer/Fen.hpp
	$(CXX) $(CXXFLAGS) $(RELEASE) -o $@ oracle/main.cpp $(ORACLE_SRC) $(LDFLAGS)

$(POSITIONS):	positions/main.cpp $(POSITIONS_SRC) positions/Positions.hpp oracle/Oracle.hpp
	$(CXX) $(CXXFLAGS) $(RELEASE) -o $@ positions/main.cpp $(POSITIONS_SRC) $(LDFLAGS)

$(EDA):	eda/main.cpp $(ORACLE_SRC) oracle/Oracle.hpp
	$(CXX) $(CXXFLAGS) $(RELEASE) -o $@ eda/main.cpp $(ORACLE_SRC) $(LDFLAGS)

$(LIB):
	$(MAKE) -C .. build/libmy_torch.a

$(GUI):	gui/main.cpp $(GUI_SRC) gui/Analysis.hpp $(LIB)
	$(CXX) $(CXXFLAGS) $(RELEASE) -pthread -o $@ gui/main.cpp $(GUI_SRC) $(LIB) $(LDFLAGS)

tests_run:	$(LIB)
	$(CXX) $(CXXFLAGS) -O2 -pthread -o $(TESTS) $(TESTS_SRC) $(POSITIONS_SRC) gui/Analysis.cpp \
		../src/analyzer/Encoding.cpp $(LIB) -lcriterion $(LDFLAGS)
	./$(TESTS)

clean:
	rm -f *.gcda *.gcno

fclean:	clean
	rm -f $(ORACLE) $(POSITIONS) $(EDA) $(GUI) $(TESTS)

re:	fclean all

.PHONY: all tests_run clean fclean re $(LIB)
