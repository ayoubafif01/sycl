all:
	nvcc -O3 -arch=sm_75 ${CUFILES} -o ${EXECUTABLE} 
clean:
	rm -f *~ *.exe