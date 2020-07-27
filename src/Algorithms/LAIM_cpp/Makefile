all: LAIM

LAIM : LAIM.o
	g++ -O3 -o LAIM LAIM.o

LAIM.o : LAIM.cpp
	g++ -O3 -c LAIM.cpp

clean :
	rm LAIM.o LAIM
