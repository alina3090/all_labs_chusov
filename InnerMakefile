ifeq ($(CXXFLAGS), )
CXXFLAGS = -O3
endif
ifeq ($(CXX), )
CXX=g++
endif

SRC=$(wildcard *.cpp)
OBJ=$(SRC:%.cpp=obj/%.o)
DEP=$(OBJ:%.o=%.d)

.phony: all clean

all:$(OUTDIR)run

$(OUTDIR)run: $(OBJ)
	$(CXX) $(CXXFLAGS) -pthread -fopenmp -ldl $^ -o $@

clean:
	rm -rf obj $(OUTDIR)run

obj/%.o:%.cpp | obj
	$(CXX) --std=c++17 -c -o $@ -MMD -MF obj/$*.d $<

obj/vector_mod.o:vector_mod.cpp | obj
	$(CXX) --std=c++20 -c -o $@ -MMD -MF obj/vector_mod.d $<

obj:
	mkdir -p obj

-include $(DEP)
