CC = gcc
CFLAGS = -ansi -Wall -Wextra -Werror -pedantic-errors
OBJ = symnmf.o

all: symnmf

symnmf: $(OBJ)
	$(CC) $(OBJ) -o symnmf $(CFLAGS) -lm

symnmf.o: symnmf.c
	$(CC) -c symnmf.c $(CFLAGS)

clean:
	rm -f *.o symnmf
