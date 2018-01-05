CC = g++

CXXFLAGS = -Wall -std=c++11 -Weffc++ -Wextra -pedantic
LFLAGS = darknet/darknet.so `pkg-config --libs opencv`

SOURCES = main.cpp
OBJ = $(SOURCES: .cpp=.o)

#rules
all: yolo

yolo: $(OBJ)
	@$(CC) -o $@ $^ $(CXXFLAGS) $(LFLAGS)

%.o: %.cpp
	@$(CC) -o $@ -c $< $(CXXFLAGS) $(LFLAGS)

clean:
	@rm yolo