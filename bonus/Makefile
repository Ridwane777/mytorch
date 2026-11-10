# Bonus (the specification requires bonus/ and allows a dedicated Makefile). The bonus reuses the FEN
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
GUI_SRC		=	gui/Analysis.cpp gui/Saliency.cpp ../src/analyzer/Encoding.cpp $(ORACLE_SRC)

# Network files are binary (ADR-0042): this prints one as text.
NN_DUMP		=	my_torch_nn_dump

# Ensembles of networks (ADR-0045): links the library and the analyzer's encodings.
ENSEMBLE	=	my_torch_ensemble

# Python module (ADR-0044): the library compiled position-independent with the pybind11 bindings.
# Not part of all: it needs pybind11, installed in python/.venv (see python/README.md).
PYTHON		?=	python/.venv/bin/python
PYMODULE	=	python/my_torch$(shell $(PYTHON) -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))" 2>/dev/null)
PY_SRC		=	python/bindings.cpp $(wildcard ../src/my_torch/*.cpp)

TESTS		=	unit_tests
TESTS_SRC	=	$(wildcard tests/*.cpp)

all:	$(ORACLE) $(POSITIONS) $(EDA) $(GUI) $(NN_DUMP) $(ENSEMBLE)

$(ORACLE):	oracle/main.cpp $(ORACLE_SRC) oracle/Oracle.hpp ../src/analyzer/Fen.hpp
	$(CXX) $(CXXFLAGS) $(RELEASE) -o $@ oracle/main.cpp $(ORACLE_SRC) $(LDFLAGS)

$(POSITIONS):	positions/main.cpp $(POSITIONS_SRC) positions/Positions.hpp oracle/Oracle.hpp
	$(CXX) $(CXXFLAGS) $(RELEASE) -o $@ positions/main.cpp $(POSITIONS_SRC) $(LDFLAGS)

$(EDA):	eda/main.cpp $(ORACLE_SRC) oracle/Oracle.hpp
	$(CXX) $(CXXFLAGS) $(RELEASE) -o $@ eda/main.cpp $(ORACLE_SRC) $(LDFLAGS)

$(LIB):
	$(MAKE) -C .. build/libmy_torch.a

$(GUI):	gui/main.cpp $(GUI_SRC) gui/Analysis.hpp gui/Saliency.hpp $(LIB)
	$(CXX) $(CXXFLAGS) $(RELEASE) -pthread -o $@ gui/main.cpp $(GUI_SRC) $(LIB) $(LDFLAGS)

$(NN_DUMP):	nn_dump/main.cpp $(LIB)
	$(CXX) $(CXXFLAGS) $(RELEASE) -pthread -o $@ nn_dump/main.cpp $(LIB) $(LDFLAGS)

$(ENSEMBLE):	ensemble/main.cpp ../src/analyzer/Encoding.cpp ../src/analyzer/Fen.cpp $(LIB)
	$(CXX) $(CXXFLAGS) $(RELEASE) -pthread -o $@ ensemble/main.cpp ../src/analyzer/Encoding.cpp \
		../src/analyzer/Fen.cpp $(LIB) $(LDFLAGS)

python:	$(PYMODULE)

# pybind11's headers are system headers here, so that -Werror applies to our code only.
$(PYMODULE):	$(PY_SRC) $(wildcard ../include/my_torch/*.hpp)
	$(CXX) $(CXXFLAGS) $(RELEASE) -pthread -fPIC -shared -fvisibility=hidden \
		$(subst -I,-isystem ,$(shell $(PYTHON) -m pybind11 --includes)) -I../src/my_torch \
		-o $@ $(PY_SRC) $(LDFLAGS)

python_tests:	$(PYMODULE)
	$(PYTHON) -m unittest discover -s python -v

tests_run:	$(LIB)
	$(CXX) $(CXXFLAGS) -O2 -pthread -o $(TESTS) $(TESTS_SRC) $(POSITIONS_SRC) gui/Analysis.cpp gui/Saliency.cpp \
		../src/analyzer/Encoding.cpp $(LIB) -lcriterion $(LDFLAGS)
	./$(TESTS)

clean:
	rm -f *.gcda *.gcno

fclean:	clean
	rm -f $(ORACLE) $(POSITIONS) $(EDA) $(GUI) $(NN_DUMP) $(ENSEMBLE) $(TESTS) python/my_torch*.so

re:	fclean all

.PHONY: all python python_tests tests_run clean fclean re $(LIB)
