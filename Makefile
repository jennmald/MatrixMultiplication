CC = cc

#Note: Change -O to -g when using debugger.
CFLAGS = -g -Wall
#CFLAGS = -O -Wall

all:	hw5_6

hw5_6.o: hw5_6.c
	$(CC) $(CFLAGS) -c $<

hw5_6: hw5_6.o
	$(CC) $< -o $@ -llapack -lblas -lm -lgfortran

plot: plotresults.m results.m
	octave -q -f --eval plotresults
	epstopdf errors.eps
	epstopdf times.eps

clean:
	rm -f hw5_6.o hw5_6
