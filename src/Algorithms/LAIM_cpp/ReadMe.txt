Welcome to use the source code of the LAIM algorithm proposed by Hongchun Wu, Jiaxing Shang, et al.


- Compile

	Makefile is included, just type type "make" to compile all the source codes.
	"gcc 4.7.2" preferred


- Execute

	Example #1 (LAIM):
		./LAIM -data NetHEPT.txt -k 50 -it 3 -theta 0.001
	Example #2 (FastLAIM):
		./LAIM -data NetHEPT.txt -k 50 -it 2 -theta 0.001 -fast

	Arguments:
		-data:
			the graph file
		-k:
			number of seed nodes
		-it:
			the iterative parameter \gamma
		-theta:
			the termination parameter (optional, default: 0.0001)
		-fast:
			if this parameter is specified, the FastLAIM algorithm will be executed


- Evalution

	To evaluate the influence spread of any algorithm using Monte-Carlo simulation, run:

		./LAIM -data NetHEPT.txt -seeds NetHEPT_seeds.txt -k 50
	
	Arguments:
		-data:
			the graph file
		-seeds:
			the seed file consisting of k lines, where each line contains the seed node id.

- Graph file format

	The first line indicates the number of nodes and edges.
	The following lines includes the source and destination node id of an edge, followed by its propagation probability

	line 1 : #nodes  #edges
	line 2 to 1+#edge : src_id  dest_id  pp

	All inputs are separated by tab(\t).

	Example:
	4	3
	0	1	0.2
	1	0	0.1
	2	3	0.4

	This graph contains four nodes and three edges. 

	Sample graph file "NetHEPT.txt" is included.

To run the program correctly, please make sure that all node ids are ranged from 0 to #nodes-1.