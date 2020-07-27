#include "Utility.h"
#include "Graph.h"
#include "MemoryUsage.h"

#include<math.h>
#include<stdlib.h>
#include<string.h>
#include<iostream>
#include<string>
#include<vector>
#include<fstream>
#include<sstream>
#include<queue>
#include<functional>
#include<set>

#define S_STATE 0
#define I_STATE 1
#define SI_STATE 2
#define R_STATE 3
#define REDUND 10

struct Pair {
	int key;
	float value;
	Pair(int key, float value) :key(key), value(value) {};
	Pair() {};
};
typedef struct Pair Pair;

void parseArg(int argn, char ** argv);
void run_laim(Graph *g, int k, int max_it, float theta);
void run_fast_laim(Graph *g, int k, int max_it, float theta);
Pair find_next(Graph *g, set<int> seed_set, int max_it, float theta);
void evaluate(Graph *g, string data, string seeds, int k);
void evaluate_total(Graph *g, string data, string seeds, int k, int simus);

float mc_influence(Graph *g, int *seed_arr, int k);
float mc_influence(Graph *g, int *seed_arr, int k, int simus);

int main(int argn, char ** argv)
{
	cout << "Program Start Successfully at: " << currentTimestampStr() << endl;
	cout << "--------------------------------------------------------------------------------" << endl;
	cout << "Arguments: ";
	for(int i = 0; i < argn; i++){
		cout << argv[i]<<" ";
	}
	cout << endl;
    parseArg( argn, argv );
    cout << "--------------------------------------------------------------------------------" << endl;
    cout<<"Program Ends Successfully at: " << currentTimestampStr() << endl;
    return 0;
}

void parseArg(int argn, char ** argv)
{
	// the parameters
    string data="";  // the path of the dataset
    int k=0;  //the # of seeds to be found
    string seeds = "";  // the path of the seed nodes for MC simulation
	bool is_total = false;
	bool run_fast = false;
	int simus = 10000;
	int max_it = 3;
	float theta = 0.001;
	vector<double> diff_param;  // the diffusion model parameters

    for(int i=0; i<argn; i++)
    {
        if(argv[i] == string("-data"))
        	data=string(argv[i+1]);
        if(argv[i] == string("-k"))
        	k=atoi(argv[i+1]);
        if(argv[i] == string("-seeds"))
        	seeds = argv[i+1];
		if(argv[i] == string("-total"))
			is_total = true;
		if(argv[i] == string("-simus"))
			simus = atoi(argv[i+1]);
		if (argv[i] == string("-it"))
			max_it = atoi(argv[i + 1]);
		if (argv[i] == string("-theta"))
			theta = atof(argv[i + 1]);
		if (argv[i] == string("-fast"))
			run_fast = true;
    }
    if (data=="")
        ExitMessage("argument data missing");
	if(k == 0)
		ExitMessage("argument k missing");
    Graph *g = new Graph(data);
	g->genProbT(WC_PARAM, diff_param);
	cout << "graph " << data << " was built!" << endl;
	if (seeds == ""){
		if(run_fast)
			run_fast_laim(g, k, max_it, theta);
		else
			run_laim(g, k, max_it, theta);
	}
    else{
		if(is_total)
			evaluate_total(g, data, seeds, k, simus);
		else
			evaluate(g, data, seeds, k);
	}
}

void run_laim(Graph *g, int k, int max_it, float theta) {
	cout << "--------------------------------------------------------------------------------" << endl;
	cout << "Start LAIM algorithm" << endl;

	clock_t time_start = clock();
	cout << "Finding top " << k << " nodes with LAIM algorithm" << endl;
	cout << "No.\tnode_id\ttime(s)\tpotential" << endl;

	int *seed_arr = new int[k];
	set<int> seed_set;
	float total_score = 0;
	for (int i = 0; i < k; i++) {
		Pair best_pair = find_next(g, seed_set, max_it+1, theta);
		seed_arr[i] = best_pair.key;
		seed_set.insert(best_pair.key);
		total_score += best_pair.value;
		cout << i + 1 << "\t" << best_pair.key << "\t" << (double)(clock() - time_start) / CLOCKS_PER_SEC << "\t" << total_score << endl;
	}
	cout << "Seeds:";
	for (int i = 0; i < k; i++) {
		cout << " " << seed_arr[i];
	}
	cout << endl;
		
	delete[] seed_arr;

	disp_mem_usage("");
	cout << "Time used: " << (double)(clock() - time_start) / CLOCKS_PER_SEC << " s" << endl;
}

void run_fast_laim(Graph *g, int k, int max_it, float theta) {
	cout << "--------------------------------------------------------------------------------" << endl;
	cout << "Start Fast LAIM algorithm" << endl;

	clock_t time_start = clock();
	cout << "Finding top " << k << " nodes with Fast LAIM algorithm" << endl;

	max_it += 1;
	double **p_arr = new double*[g->n];
	for (int i = 0; i < g->n; i++) {
		double *p = new double[max_it + 1];
		memset(p, 0, (max_it + 1) * sizeof(double));
		p[1] = 1;
		p_arr[i] = p;
	}
	int it = 2;
	while (it < max_it) {
		for (int i = 0; i < g->n; i++) {
			double *p = p_arr[i];
			int k_out = g->gT[i].size();
			for (int j = 0; j < k_out; j++) {
				int neigh = g->gT[i][j];
				double w1 = 1.0 / k_out;
				double w2 = g->probT[i][j];
				double potential_1 = p_arr[neigh][it - 1];
				double potential_2 = p[it - 2];
				if (potential_1 > theta && potential_1 - w1 * potential_2 > theta)
					p[it] = p[it] + w2 * (potential_1 - w1 * potential_2);
			}
			p[max_it] = p[max_it] + p[it];
		}
		it++;
	}

	cout << "No.\tnode_id\ttime(s)\tpotential" << endl;
	int *seed_arr = new int[k];
	float total_score = 0;
	for (int i = 0; i < k; i++){
		double max = 0;
		int new_seed = -1;
		for(int j = 0; j < g->n; j++){
			if(p_arr[j][max_it] > max){
				max = p_arr[j][max_it];
				new_seed = j;
			}
		}
		seed_arr[i] = new_seed;
		total_score += max;
		p_arr[new_seed][max_it] = 0;
		cout << i + 1 << "\t" << new_seed << "\t" << (double)(clock() - time_start) / CLOCKS_PER_SEC << "\t" << total_score << endl;
	}
	cout << "Seeds:";
	for (int i = 0; i < k; i++) {
		cout << " " << seed_arr[i];
	}
	cout << endl;

	disp_mem_usage("");
	cout << "Time used: " << (double)(clock() - time_start) / CLOCKS_PER_SEC << " s" << endl;

	for (int i = 0; i < g->n; i++) {
		delete[] p_arr[i];
	}
	delete[] p_arr;
	delete[] seed_arr;
}

Pair find_next(Graph *g, set<int> seed_set, int max_it, float theta) {
	bool *is_seed = new bool[g->n];
	memset(is_seed, false, g->n * sizeof(bool));
	set<int>::iterator iter = seed_set.begin();
	while(iter != seed_set.end()){
		is_seed[*iter] = true;
		iter++;
	}

	double **p_arr = new double*[g->n];
	for (int i = 0; i < g->n; i++) {
		double *p = new double[max_it + 1];
		memset(p, 0, (max_it + 1) * sizeof(double));
		if (!is_seed[i]) {
			p[1] = 1;
			p[max_it] = 1;
		}
		p_arr[i] = p;
	}
	int count = 0;
	int it = 2;
	while (it < max_it && count < g->n) {
		count = 0;  //the number of nodes whose potential is updated in each iteration
		for (int i = 0; i < g->n; i++) {
			if (is_seed[i]) {
				count++;
				continue;
			}
			double *p = p_arr[i];
			int k_out = g->gT[i].size();
			for (int j = 0; j < k_out; j++) {
				int neigh = g->gT[i][j];
				double w1 = 1.0 / k_out;
				double w2 = g->probT[i][j];
				double potential_1 = p_arr[neigh][it - 1];
				double potential_2 = p[it - 2];
				if (potential_1 > theta && potential_1 - w1 * potential_2 > theta)
					p[it] = p[it] + w2 * (potential_1 - w1 * potential_2);
			}
			p[max_it] = p[max_it] + p[it];
			if (p[it] <= theta)
				count++;
		}
		it++;
	}
	double max = 0;
	double new_seed = -1;
	for (int i = 0; i < g->n; i++) {
		if (p_arr[i][max_it] > max) {
			max = p_arr[i][max_it];
			new_seed = i;
		}
	}
	Pair p(new_seed, max);
	for (int i = 0; i < g->n; i++) {
		delete[] p_arr[i];
	}
	delete[] p_arr;
	delete[] is_seed;
	return p;
}

void evaluate(Graph *g, string data, string seeds, int k) {
	cout << "evaluating influence... data:" << data << " seeds:" << seeds << endl;
	int *seed_arr = new int[k];
	ifstream ifs(seeds.c_str());
	if (!ifs)
		cout << "seeds file: " << seeds << " cannot be openned!" << endl;
	else
		cout << "seeds file: " << seeds << " openned!" << endl;
	cout << "id\tseed\tinfluence\ttimestamp" << endl;
	string buffer;
	int point_arr[11] = { 1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50 };
	float *inf_arr = new float[11];
	int id = 0;
	for (int i = 0; i < k; i++) {
		ifs >> buffer;
		seed_arr[i] = atoi(buffer.c_str());
		int match = 0;
		for (int j = 0; j < 11; j++) {
			if (point_arr[j] == i + 1) {
				match = 1;
				break;
			}
		}
		if (match) {
			float inf = mc_influence(g, seed_arr, i + 1);
			inf_arr[id++] = inf;
			cout << i + 1 << "\t" << seed_arr[i] << "\t";
			cout << inf << '\t' << currentTimestampStr() << endl;
		}
	}
	cout << "inf=[" << inf_arr[0];
	for (int i = 1; i < id; i++) {
		cout << ", " << inf_arr[i];
	}
	cout << "];" << endl;
	delete[] inf_arr;
}

float mc_influence(Graph *g, int *seed_arr, int k){
	srand((unsigned)time(NULL));
	double inf = 0;
	int *i_arr = new int[g->n]; //the array of current active nodes
	int i_size = 0; // the # of newly active nodes 
	int *r_arr = new int[g->n]; // the array of previous active nodes
	int r_size = 0; // the # of previously active nodes
	int *si_arr = new int[g->n];  // the array of nodes to be active in t+1
	int si_size = 0; // the # of nodes to be active in t+1
	int *state_arr = new int[g->n]; // the state of nodes
	memset(state_arr, S_STATE, g->n * sizeof(int)); // initialize the state array	
	int *rand_arr = new int[g->n]; //the 0 ~ n-1 numbers sorted by random order
	for(int r = 0; r < NUM_SIMUS; r++){
		double active_size = 0;
		//reset the state of all nodes		
		for(int i = 0; i < r_size; i++){
			state_arr[r_arr[i]] = S_STATE;
		}		
		r_size = 0;		
		// initialize the seed nodes
		for(int i = 0; i < k; i++){
			i_arr[i_size++] = seed_arr[i];
			state_arr[i_arr[i]] = I_STATE;
		}
		while(i_size > 0){
			active_size += i_size;
			si_size = 0;
			randomOrder(rand_arr, i_size);
			for(int i = 0; i < i_size; i++){
				int i_node = i_arr[i];
				int k_out = g->gT[i_node].size();
				for(int j = 0; j < k_out; j++){
					int neigh = g->gT[i_node][j];
					if (state_arr[neigh] == S_STATE) {
						double pp = g->probT[i_node][j];
						double rand_float = ((double)rand()) / RAND_MAX;
						if(rand_float < pp) {
							state_arr[neigh] = SI_STATE;
							si_arr[si_size++] = neigh;
						}
					}					
				}
			}
			for(int i = 0; i < i_size; i++){
				state_arr[i_arr[i]] = R_STATE;
				r_arr[r_size++] = i_arr[i];
			}
			i_size = 0;
			for(int i = 0; i < si_size; i++){
				state_arr[si_arr[i]] = I_STATE;
				i_arr[i_size++] = si_arr[i];
			}
		}
		inf += active_size;
	}
	delete[] i_arr;
	delete[] r_arr;
	delete[] si_arr;
	delete[] state_arr;
	delete[] rand_arr;
	return inf / NUM_SIMUS;
}

void evaluate_total(Graph *g, string data, string seeds, int k, int R){
	cout << "evaluating overall influence... data:" << data << " seeds:" << seeds << endl;
	int *seed_arr = new int[k];
	ifstream ifs(seeds.c_str());
	if(!ifs)
		cout << "seeds file: " << seeds << " cannot be openned!" << endl;
	else
		cout << "seeds file: " << seeds << " openned!" << endl;
	string buffer;
	cout << "Seeds:";
	for(int i = 0; i < k; i++){
		ifs >> buffer;
		seed_arr[i] = atoi(buffer.c_str());
		cout << " " << seed_arr[i];
	}
	cout << endl;
	float total_inf = mc_influence(g, seed_arr, k, R);
	cout << "Total influence: " << total_inf << endl;
}

float mc_influence(Graph *g, int *seed_arr, int k, int simus){
	srand((unsigned)time(NULL));
	double inf = 0;
	int *i_arr = new int[g->n]; //the array of current active nodes
	int i_size = 0; // the # of newly active nodes 
	int *r_arr = new int[g->n]; // the array of previous active nodes
	int r_size = 0; // the # of previously active nodes
	int *si_arr = new int[g->n];  // the array of nodes to be active in t+1
	int si_size = 0; // the # of nodes to be active in t+1
	int *state_arr = new int[g->n]; // the state of nodes
	memset(state_arr, S_STATE, g->n * sizeof(int)); // initialize the state array	
	int *rand_arr = new int[g->n]; //the 0 ~ n-1 numbers sorted by random order
	for(int r = 0; r < simus; r++){
		double active_size = 0;
		//reset the state of all nodes		
		for(int i = 0; i < r_size; i++){
			state_arr[r_arr[i]] = S_STATE;
		}		
		r_size = 0;		
		// initialize the seed nodes
		for(int i = 0; i < k; i++){
			i_arr[i_size++] = seed_arr[i];
			state_arr[i_arr[i]] = I_STATE;
		}
		while(i_size > 0){
			active_size += i_size;
			si_size = 0;
			randomOrder(rand_arr, i_size);
			for(int i = 0; i < i_size; i++){
				int i_node = i_arr[i];
				int k_out = g->gT[i_node].size();
				for(int j = 0; j < k_out; j++){
					int neigh = g->gT[i_node][j];
					if (state_arr[neigh] == S_STATE) {
						double pp = g->probT[i_node][j];
						double rand_float = ((double)rand()) / RAND_MAX;
						if(rand_float < pp) {
							state_arr[neigh] = SI_STATE;
							si_arr[si_size++] = neigh;
						}
					}					
				}
			}
			for(int i = 0; i < i_size; i++){
				state_arr[i_arr[i]] = R_STATE;
				r_arr[r_size++] = i_arr[i];
			}
			i_size = 0;
			for(int i = 0; i < si_size; i++){
				state_arr[si_arr[i]] = I_STATE;
				i_arr[i_size++] = si_arr[i];
			}
		}
		inf += active_size;
	}
	delete[] i_arr;
	delete[] r_arr;
	delete[] si_arr;
	delete[] state_arr;
	delete[] rand_arr;
	return inf / simus;
}