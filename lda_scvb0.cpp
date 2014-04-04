#include <vector>
#include <limits>
#include <algorithm>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <assert.h>
#include <omp.h>
#include <iostream>
#include <sstream>
#include <fstream>
#include <numeric>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <tclap/CmdLine.h>

using namespace std;
using namespace Eigen;
using namespace TCLAP;


// pair comparer copied from shaofeng mo
bool pairCmp( const pair<int,double>& p1, const pair<int,double>& p2){
  if( p1.second - p2.second > 0.00000000001 ){
    return true;
  }else{
    return false;
  }
}


struct Document {
    vector<int> word_ind;
    vector<int> word_count;
};


Document* get_str2doc(string str) 
{
    // create Document struct from one-line str, return pointer
    Document * doc = new Document;
	//NOTE: istringstream is ambiguous with tclap; compiling error w/o std scoping
    std::istringstream iss(str);
    int w_i, w_c; 
    while (iss >> w_i >> w_c) 
    {
        doc->word_ind.push_back(w_i);
        doc->word_count.push_back(w_c);
    }
    /*for (int i=0; i<doc->word_ind.size(); i++)
    {
        cerr << "w_i:" << doc->word_ind[i] << "\tw_c:" << doc->word_count[i] << endl;
        
    }*/
    return doc;
}


void read_vocab(string file, vector<string> & vocabs) 
{
    clock_t t_b4_read = clock();
    vocabs.push_back("<unk>");    

    ifstream ifs(file.c_str());
    for (string line ; getline(ifs, line); )
    {
       vocabs.push_back(line); 
    }
    cerr << "done reading vocab file, time spent: " << float(clock() - t_b4_read)/CLOCKS_PER_SEC << " secs" << endl << endl;
}


Document** read_docword(string file, int & D, int & W, int & C) 
{
    clock_t t_b4_read = clock();

    Document** documents;
    ifstream ifs(file.c_str());
    int line_count = 0;
    for (string line ; getline(ifs, line); )
    {
        //cerr << "line*"<<line<<"*\n";
        line_count += 1;
        if (line_count % 100000 == 0) cerr << "reading " << line_count << " lines" << endl;
        if (line_count == 1) 
        {
            D = atoi(line.c_str());
            documents = new Document* [D];
            for (int counter = 0; counter < D; counter++)
            {
                documents[counter] = new Document;
            }
        }
        else if (line_count == 2)
        {
            //NOTE: different from uci bag-of-words sparse format, word index w_i is 0-based and 0 is reserved for unknown words <unk> potentially existing in test doc, therefore +1
            W = atoi(line.c_str()) + 1;
        }
        else if (line_count != 3) 
        {
			//NOTE: istringstream is ambiguous with tclap; compiling error w/o std scoping
            std::istringstream iss(line);
            int d_j, w_i, w_c;
            iss >> d_j >> w_i >> w_c;
            C += w_c;
            d_j -= 1;
            //cerr << d_j << " " << w_i << " " << w_c << endl;
            documents[d_j]->word_ind.push_back(w_i);
            documents[d_j]->word_count.push_back(w_c);
        }
    }
    cerr << "D:" << D << "\tW:" << W << "(adding word index 0 for unknown word type <unk>)\tC:" << C << endl; 
    cerr << "done reading docword file, time spent: " << float(clock() - t_b4_read)/CLOCKS_PER_SEC << " secs" << endl << endl;
    return documents;
}


double get_perplexity(Document ** documents,
                      const int & D,
                      const int & C,
                      const MatrixXd & mat_posterior_prob_phi, 
                      const MatrixXd & mat_posterior_prob_theta)
{
    double total_log_word_prob = 0.0;
    for (int j=0; j<D; j++)
    {
        Eigen::initParallel();
        int num_unique_words = documents[j]->word_ind.size(); 
        #pragma omp parallel for firstprivate(num_unique_words) reduction(+:total_log_word_prob)
        for (int i=0; i<num_unique_words; i++)
        {
            int w_i = documents[j]->word_ind[i];
            int w_c = documents[j]->word_count[i];
            double word_prob = mat_posterior_prob_phi.row(w_i).dot(mat_posterior_prob_theta.row(j));
            double log_word_prob_c = log2(word_prob) * w_c;
            total_log_word_prob += log_word_prob_c;        
        }
    } 
    double perplexity = exp2(-total_log_word_prob/C);
    return perplexity;
}


void get_mat_posterior_prob_phi(const MatrixXd & mat_N_phi, 
                                const double & ETA,
                                MatrixXd & mat_posterior_prob_phi)
{
    MatrixXd mat_N_phi_adjusted = (mat_N_phi.array() + ETA).matrix();
    MatrixXd mat_sum_col = mat_N_phi_adjusted.colwise().sum();
    for (int c=0; c<mat_N_phi_adjusted.cols(); c++)
    {   
        mat_posterior_prob_phi.col(c) = mat_N_phi_adjusted.col(c) / mat_sum_col(0, c);
    }
    //cerr << "probphi " << endl << mat_posterior_prob_phi << endl;
}


void get_mat_posterior_prob_theta(const MatrixXd & mat_N_theta, 
                                  const double & ALPHA,
                                  MatrixXd & mat_posterior_prob_theta)
{
    MatrixXd mat_N_theta_adjusted = (mat_N_theta.array() + ALPHA).matrix();
    MatrixXd mat_sum_row = mat_N_theta_adjusted.rowwise().sum();
    for (int r=0; r<mat_N_theta_adjusted.rows(); r++)
    {   
        mat_posterior_prob_theta.row(r) = mat_N_theta_adjusted.row(r) / mat_sum_row(r, 0);
    }
    //cerr << "probtheta " << endl << mat_posterior_prob_theta << endl;
}


//TODO: improve efficiency, don't use temporary MatrixXd object as function arg
MatrixXd burnin_doc_j(const Document * doc_j,
                      const int & W,
                      const int & K,
                      const double & ALPHA,
                      const double & ETA,
                      const int & BURNIN_PER_DOC,
                      const double & S2,
                      const double & TAO2,
                      const double & KAPPA2,
                      const MatrixXd & mat_N_phi,
                      MatrixXd mat_N_theta_j)
{
    //NOTE: used vectorized update learnt from michael hankin, a different way of clumping from jimmy foulds's orig paper. helps perplexity a lot.    

    // compute doc length
    int C_j = 0;
    for (int i=0;i<doc_j->word_count.size();i++)
    {
        C_j += doc_j->word_count[i];
    }
    
    // create randomly shuffled indices, helps marginally
    vector<int> rand_indices;
    for (int p = 0; p < doc_j->word_ind.size(); ++p)
        rand_indices.push_back(p);
    random_shuffle(rand_indices.begin(), rand_indices.end());
    //cerr << "shuffled ind for " << j << " ";
    //for (int si=0; si<rand_indices.size(); si++) cerr << rand_indices[si] << " ";
    //cerr << endl;
        
    MatrixXd mat_N_Z = mat_N_phi.colwise().sum();
    for (int b=0; b<BURNIN_PER_DOC; b++)
    {
        double rho_theta = S2 / pow(TAO2+b, KAPPA2);
        double weighted_leftstep = 0.0;
        MatrixXd weighted_mat_gamma_i_j = MatrixXd::Constant(1, K, 0.0);
        
        for (int q=0; q<rand_indices.size(); q++)
        {
            int i = rand_indices[q];
            int w_i = doc_j->word_ind[i];
            int w_c = doc_j->word_count[i];
            //cerr << "i " << i << " w_i " << w_i << " w_c " << w_c << endl;
    
            // eqn 5
            MatrixXd mat_gamma_i_j = (((mat_N_phi.row(w_i).array() + ETA)*(mat_N_theta_j.array() + ALPHA))/( mat_N_Z.array() + ETA*W)).matrix();
            mat_gamma_i_j /= mat_gamma_i_j.sum(); 
            //cerr << "mat_gamma_i_j " << mat_gamma_i_j << endl;
            
            // eqn 9 modified
            weighted_leftstep += pow(1-rho_theta, w_c)*w_c/C_j;
            weighted_mat_gamma_i_j += (1-pow(1-rho_theta, w_c))*w_c*mat_gamma_i_j;
        }
        mat_N_theta_j = weighted_leftstep * mat_N_theta_j + weighted_mat_gamma_i_j;
    }
    //cerr << "mat_N_theta_j after burnin " << mat_N_theta_j << endl;
    return mat_N_theta_j;
}


// predict doc in training set that is most similar to testdoc
int get_similar_by_l2_distance(const Document * testdoc, 
                                const int & D,
                                const int & W,
                                const int & K,
                                const double & ALPHA,
                                const double & ETA,
                                const int & BURNIN_PER_DOC,
                                const double & S2,
                                const double & TAO2,
                                const double & KAPPA2,
                                const MatrixXd & mat_N_phi,
                                const MatrixXd & mat_posterior_prob_theta)
{
    MatrixXd mat_N_theta_test = ((MatrixXd::Random(1, K)*0.5).array() + 1).matrix();
    mat_N_theta_test = burnin_doc_j(testdoc, W, K, ALPHA, ETA, BURNIN_PER_DOC, S2, TAO2, KAPPA2, mat_N_phi, mat_N_theta_test);
    MatrixXd mat_posterior_prob_theta_test = MatrixXd::Constant(1, K, 0);
    get_mat_posterior_prob_theta(mat_N_theta_test, ALPHA, mat_posterior_prob_theta_test);
   
    int best_doc_ind = 0;
    double best_dist = std::numeric_limits<long double>::infinity(); 
    for (int j=0; j<D; j++)
    {
        double dist = (mat_posterior_prob_theta_test - mat_posterior_prob_theta.row(j)).norm();
        if (dist < best_dist) 
        {
            best_doc_ind = j; 
            best_dist = dist;
        } 
    }
    return best_doc_ind;
}


int get_similar_by_condprob(const Document * testdoc, 
                            const MatrixXd & mat_posterior_prob_phi,
                            const MatrixXd & mat_posterior_prob_theta,
                            const int & D)
{
    // given testdoc, predict the most similar doc we have in training
    // implementing eqn 9 in "Probabilistic Topic Models" by Steyvers and Griffiths
    int best_doc_ind = 0;
    double best_log_doc_prob = -std::numeric_limits<long double>::infinity(); 
    for (int j=0; j<D; j++)
    {
        int num_unique_words = testdoc->word_ind.size();
        double log_doc_prob = 0.0; 
        for (int i=0; i<num_unique_words; i++)
        {
            int w_i = testdoc->word_ind[i];
            int w_c = testdoc->word_count[i];
            double word_prob = mat_posterior_prob_phi.row(w_i).dot(mat_posterior_prob_theta.row(j));
            double log_word_prob_c = log2(word_prob) * w_c;
            log_doc_prob += log_word_prob_c;        
        }
        if (log_doc_prob > best_log_doc_prob) 
        {
            best_doc_ind = j; 
            best_log_doc_prob = log_doc_prob;
        } 
    }
    return best_doc_ind;
}


void scvb0(Document ** documents, 
          const int & D,
          const int & W,
          const int & C,
          const int & K,
          const int & SWEEP,
          const double & ALPHA,
          const double & ETA,
          const int & M,
          const int & BURNIN_PER_DOC,
          const double & S,
          const double & TAO,
          const double & KAPPA,
          const double & S2,
          const double & TAO2,
          const double & KAPPA2,
          const int & RANDSEED_phi,
          const int & RANDSEED_theta,
          const int & TIME_LIMIT,
          const int & REPORT_PERPLEXITY_ITER,
          MatrixXd & mat_N_phi,
          MatrixXd & mat_N_theta,
          MatrixXd & mat_posterior_prob_phi,
          MatrixXd & mat_posterior_prob_theta)
{
    bool debug = false;
    cerr << "start scvb0" << endl;
    clock_t t_b4_scvb0 = clock();

    //randomization is key, yet how to randomize and whether or not to normalize is not important 
    cerr << "initializing..." << endl;
    clock_t t_b4_init = clock();
    mat_N_phi = ((MatrixXd::Random(W, K)*0.5).array() + 1).matrix();
    mat_N_phi = mat_N_phi*float(C)/mat_N_phi.sum();
    mat_N_theta = ((MatrixXd::Random(D, K)*0.5).array() + 1).matrix();
    mat_N_theta = mat_N_theta*float(C)/mat_N_theta.sum();
    mat_posterior_prob_phi = MatrixXd::Constant(W, K, 0);
    mat_posterior_prob_theta = MatrixXd::Constant(D, K, 0);
    cerr << "done initializing, time spent: " << float(clock() - t_b4_init)/CLOCKS_PER_SEC << " secs" << endl;

    double minibatches_per_corpus = (1.0 * C) / M;
    unsigned long number_minibatches = ceil(float(D)/M);

    for (int s=0; s<SWEEP; s++)
    {
        clock_t t_b4_sweep = clock();
        #pragma omp parallel for
        for (int m=0; m<number_minibatches; m++)
        {
            MatrixXd mat_N_phi_hat = MatrixXd::Constant(W, K, 0);
            MatrixXd mat_N_Z_hat = MatrixXd::Constant(1, K, 0);
            MatrixXd mat_N_Z = mat_N_phi.colwise().sum();

            int minibatch_start = m*M;
            int minibatch_end = ( (m+1)*M > D ? D : (m+1)*M );
            unsigned long doc_iter;

            //cerr << "batch " << m << " start " << minibatch_start << " end " <<  minibatch_end << endl;
            for (int j=minibatch_start; j<minibatch_end; j++)
            {
                doc_iter = s*D + j + 1;
                // burnin: update j-th row of mat_N_theta
                mat_N_theta.row(j) = burnin_doc_j(documents[j], W, K, ALPHA, ETA, BURNIN_PER_DOC, S2, TAO2, KAPPA2, mat_N_phi, mat_N_theta.row(j));
                
                // final update for doc
                for (int i=0; i<documents[j]->word_ind.size(); i++)
                {
                    int w_i = documents[j]->word_ind[i];
                    int w_c = documents[j]->word_count[i];
                    // eqn 5
                    MatrixXd mat_gamma_i_j = (((mat_N_phi.row(w_i).array() + ETA)*(mat_N_theta.row(j).array() + ALPHA))/( mat_N_Z.array() + ETA*W)).matrix();
                    mat_gamma_i_j /= mat_gamma_i_j.sum(); 
        
                    // update hat            
                    mat_N_phi_hat.row(w_i) += minibatches_per_corpus*w_c*mat_gamma_i_j;
                    mat_N_Z_hat += minibatches_per_corpus*w_c*mat_gamma_i_j;
                }

                /* 
                // report perplexity for certain number of doc iterations
                if (doc_iter % REPORT_PERPLEXITY_ITER == 0)
                {
                    get_mat_posterior_prob_phi(mat_N_phi, ETA, mat_posterior_prob_phi);
                    get_mat_posterior_prob_theta(mat_N_theta, ALPHA, mat_posterior_prob_theta);
                    double perplexity = get_perplexity(documents, D, C, mat_posterior_prob_phi, mat_posterior_prob_theta);
                    //cerr << "doc_iter " << doc_iter << " perplexity " << perplexity << endl;
                }*/

            } // finish doc update

            // batch update
            double rho_phi = S/pow(TAO+doc_iter, KAPPA);
            // eqn 7
            mat_N_phi = (1-rho_phi)*mat_N_phi + rho_phi*mat_N_phi_hat;
            // eqn 8
            mat_N_Z = (1-rho_phi)*mat_N_Z + rho_phi*mat_N_Z_hat;
        } // finish minibatch update
        
        // report perplexity after finishing a sweep
        get_mat_posterior_prob_phi(mat_N_phi, ETA, mat_posterior_prob_phi);
        get_mat_posterior_prob_theta(mat_N_theta, ALPHA, mat_posterior_prob_theta);
        double perplexity = get_perplexity(documents, D, C, mat_posterior_prob_phi, mat_posterior_prob_theta);
        clock_t t_after_sweep = clock();
        cerr << "done sweep " << s << ", time spent: " << float(t_after_sweep - t_b4_sweep)/CLOCKS_PER_SEC << " secs, perplexity: " << perplexity << endl;
        
        // check timeout!
        if ( float(t_after_sweep - t_b4_scvb0)/CLOCKS_PER_SEC > TIME_LIMIT) 
        {
            cerr << "reaching time limit " << TIME_LIMIT << " secs, stop scvb0" << endl;
            return;
        }

    } // finish sweep 
    cerr << "done scvb0, time spent: " << float(clock() - t_b4_scvb0)/CLOCKS_PER_SEC << " secs" << endl << endl;
}


int main( int argc,      // Number of strings in array argv
          char *argv[],   // Array of command-line argument strings
          char *envp[] )  // Array of environment variable strings
{
    //speed up c++ io stream
    std::ios::sync_with_stdio(false);

    //parse command line args; if succeed, perform scvb0 training (and prediction with -p switch is on)
    try {  
        CmdLine cmd("Command description message", ' ', "0.2");

        ValueArg<string> docwordInFileArg("d","docwordfile","Path to docword file in uci sparse bag-of-words format, where each line is a (doc id, word id, word count) triple delimited by space. Both doc id and word id are 1-based. Word id refers to the corresponding line in the vocab file", true, "","string");
        cmd.add( docwordInFileArg );

        ValueArg<string> vocabInFileArg("v","vocabfile","Path to vocab file in uci sparse bag-of-words format, note that word index 0 is internally reserved for unknown words <unk> potentially existing in test doc", true, "","string");
        cmd.add( vocabInFileArg );

        ValueArg<string> doctopicOutFileArg("","doctopicfile","Output result of scvb0 training, where each line represents the distribution of all topics for a document, separated by commas", false, "","string");
        cmd.add( doctopicOutFileArg );

        ValueArg<string> topicwordOutFileArg("","topicwordfile","Output result of scvb0 training, where each line lists for a topic all word ids sorted by their probabilities in a descending order. If there are 1000 word types (i.e., 1000 lines in the vocab file), this line will have 1001 entries in the format 'wordID:weight, wordID:weight, ...wordID:weight', where wordID=0 is reserved for unknown words <unk> which may appear in query document", false, "","string");
        cmd.add( topicwordOutFileArg );

        ValueArg<string> topicwordOutFile2Arg("","topicwordfile2","Same as --topicwordfile, yet for better human consumption only lists the top 100 words and replaces word id by word type", false, "","string");
        cmd.add( topicwordOutFile2Arg );

        ValueArg<int> sweepArg("s","sweep","Number of sweeps over docword file", false, 10,"int");
        cmd.add( sweepArg );

        ValueArg<int> numTopicArg("k","ktopics","Number of topics for LDA model", false, 100,"int");
        cmd.add( numTopicArg );

        ValueArg<double> alphaArg("a","alpha","Dirichlet prior for doc-topic distribution", false, 0.1,"double");
        cmd.add( alphaArg );

        ValueArg<double> etaArg("e","eta","Dirichlet prior for topic-word distribution", false, 0.01,"double");
        cmd.add( etaArg );

        ValueArg<int> randseedPhiArg("","rsphi","Seed for randomized initialization of N_phi matrix", false, 9,"int");
        cmd.add( randseedPhiArg );

        ValueArg<int> randseedThetaArg("","rstheta","Seed for randomized initialization of N_theta matrix", false, 1119,"int");
        cmd.add( randseedThetaArg );

        ValueArg<int> minibatchArg("m","minibatchsize","Size of each minibatch for training docs", false, 20,"int");
        cmd.add( minibatchArg );

        ValueArg<int> burninPerDocArg("b","burnin","Number of burn-ins for each training doc", false, 15,"int");
        cmd.add( burninPerDocArg );

        ValueArg<double> sArg("","scale","Scale to compute stepsize for doc iteration update", false, 10.0,"double");
        cmd.add( sArg );

        ValueArg<double> taoArg("","tao","Tao to compute stepsize for doc iteration update", false, 1000.0,"double");
        cmd.add( taoArg );

        ValueArg<double> kappaArg("","kappa","Kappa to compute stepsize for doc iteration update", false, 0.9,"double");
        cmd.add( kappaArg );

        ValueArg<double> s2Arg("","scale2","Scale2 to compute stepsize for doc iteration update", false, 50.0,"double");
        cmd.add( s2Arg );

        ValueArg<double> tao2Arg("","tao2","Tao2 to compute stepsize for doc iteration update", false, 105.0,"double");
        cmd.add( tao2Arg );

        ValueArg<double> kappa2Arg("","kappa2","Kappa2 to compute stepsize for doc iteration update", false, 0.9,"double");
        cmd.add( kappa2Arg );

        ValueArg<int> timeLimitArg("","timelimit","Time limit for training, in seconds", false, 36000,"int");
        cmd.add( timeLimitArg );

        ValueArg<int> reportPerpIterArg("","reportperp","Report training perplexity for how many document iterations", false, 1000,"int");
        cmd.add( reportPerpIterArg );

        ValueArg<string> similarityArg("","similarity","Method to measure similarity between the query and training docs, default to L2 distance of topic distribution. Another similarity measure, P(query_doc|training_doc) requires more computation and can be turned on by specifying --similarity condprob", false, "l2","string");
        cmd.add( similarityArg );

        SwitchArg predictSimilarSwitch("p","predict","optional prediction mode, i.e., after training topic model, it accepts a one-line compact docword representation from STDIN and outputs to STDOUT the id of the most similar document (1-based) from the training set", false);
        cmd.add( predictSimilarSwitch );

        //TODO: function to specify random seed explicitly
        SwitchArg debugSwitch("","debug","running debug mode, specifying random seed and sanity check to see if each training doc predicts itself as the most similar doc", false);
        cmd.add( debugSwitch );

        cmd.parse( argc, argv );

        // Get the value parsed by each arg. 
        const string f_docword = docwordInFileArg.getValue();
        const string f_vocab = vocabInFileArg.getValue();
        const string f_doctopic = doctopicOutFileArg.getValue();
        const string f_topicword = topicwordOutFileArg.getValue();
        const string f_topicword2 = topicwordOutFile2Arg.getValue();
        const string similarity = similarityArg.getValue();

        const int SWEEP = sweepArg.getValue();
        const int K = numTopicArg.getValue();

        const double ALPHA=alphaArg.getValue();
        const double ETA=etaArg.getValue();

        const int RANDSEED_phi = randseedPhiArg.getValue();
        const int RANDSEED_theta = randseedThetaArg.getValue();
        const int M= minibatchArg.getValue();
        const int BURNIN_PER_DOC = burninPerDocArg.getValue();

        const double S=sArg.getValue();
        const double TAO=taoArg.getValue();
        const double KAPPA=kappaArg.getValue();

        const double S2=s2Arg.getValue();
        const double TAO2=tao2Arg.getValue();
        const double KAPPA2=kappa2Arg.getValue();

        const int TIME_LIMIT = timeLimitArg.getValue();
        const int REPORT_PERPLEXITY_ITER = reportPerpIterArg.getValue();

        const bool predictSimilar = predictSimilarSwitch.getValue();
        const bool debug = debugSwitch.getValue();

        // report parameters
        cerr << "\nParameters:\n";

        cerr << "Input docword file: " << f_docword << endl;
        cerr << "Input vocab file: " << f_vocab << endl;
        cerr << "Output doctopic file: " << f_doctopic << endl;
        cerr << "Output topicword file: " << f_topicword << endl;
        cerr << "Output topicword2 file: " << f_topicword2 << endl;

        cerr << "K: " << K << "\tSWEEP: " << SWEEP << endl;
        cerr << "ALPHA: " << ALPHA << "\tETA: " << ETA << endl;
        cerr << "M: " << M << "\tBURNIN_PER_DOC: " << BURNIN_PER_DOC << endl;
        cerr << "S: " << S << "\tTAO: " << TAO << "\tKAPPA: " << KAPPA << endl; 	
        cerr << "S2: " << S2 << "\tTAO2: " << TAO2 << "\tKAPPA2: " << KAPPA2 << endl; 	
        cerr << "RANDSEED_phi: " << RANDSEED_phi << "\tRANDSEED_theta: " << RANDSEED_theta << endl;
        cerr << "TIME_LIMIT: " << TIME_LIMIT << "\tREPORT_PERPLEXITY_ITER: " << REPORT_PERPLEXITY_ITER << endl;
        cerr << endl;

        // read vocab file
        vector<string> vocabs;
        read_vocab(f_vocab, vocabs);
        //for (int i=0; i<vocabs.size(); i++) cerr << vocabs[i] << endl;
        //cerr << "total vocab " << vocabs.size() << endl;

        // read docword file and obtain D, W, C
        int D=0, W=0, C=0;
        //NOTE: error if we pass this pointer as function arg
        //Document** documents = NULL;
        Document** documents = read_docword(f_docword, D, W, C); 
        //TODO: assert to check size consistency of docword file and vocab file
        //assert (vocabs.size()==W);
   
        // report empty doc to stdout, convert internal 0-based doc ind to 1-based
        int empty_doc_count = 0;
        for (int d=0; d<D; d++)
        {
            if (documents[d]->word_ind.size() == 0)
            { 
                cerr << "empty doc: " << d+1 << endl;
                empty_doc_count++;
            }
        }
        cerr << "# of empty doc: " << empty_doc_count << endl << endl;

        // scvb0
        MatrixXd mat_N_phi, mat_N_theta, mat_posterior_prob_phi, mat_posterior_prob_theta;
        scvb0(documents, 
              D,
              W,
              C,
              K,
              SWEEP,
              ALPHA,
              ETA,
              M,
              BURNIN_PER_DOC,
              S,
              TAO,
              KAPPA,
              S2,
              TAO2,
              KAPPA2,
              RANDSEED_phi,
              RANDSEED_theta,
              TIME_LIMIT,
              REPORT_PERPLEXITY_ITER,
              mat_N_phi,
              mat_N_theta,
              mat_posterior_prob_phi,
              mat_posterior_prob_theta);

        // output results to files
        if (f_doctopic != "")
        {
            IOFormat matrixFormat(FullPrecision, DontAlignCols, ", ", "\n", "", "", "", "");
            cerr << "writing to doctopic file: " << f_doctopic << endl;
            clock_t t_b4_write = clock();
            ofstream ofs;
            ofs.open(f_doctopic.c_str());
            ofs << mat_posterior_prob_theta.format(matrixFormat) << endl;
            ofs.close();
            cerr << "done, time spent: " << float(clock() - t_b4_write)/CLOCKS_PER_SEC << " secs" << endl << endl;
        } 
        if (f_topicword != "")
        {
            cerr << "writing to topicword file: " << f_topicword << endl;
            clock_t t_b4_write = clock();
            ofstream ofs;
            ofs.open(f_topicword.c_str());
            for (int c=0; c<mat_posterior_prob_phi.cols(); c++)
            {
                // create vector for each column in matrix
                vector< pair<int, double> > pair_ind_prob;
                for (int r=0; r<mat_posterior_prob_phi.rows(); r++)
                {
                    pair_ind_prob.push_back( pair<int, double> (r, mat_posterior_prob_phi(r, c)) );
                }
                // sort vector by descending order, keeping track of indices
                sort( pair_ind_prob.begin() , pair_ind_prob.end() , pairCmp );

                // output all word ids with probability for each topic
                for (int n=0; n< pair_ind_prob.size() ; n++) 
                {
                    ofs << pair_ind_prob[n].first << ":" << pair_ind_prob[n].second;
                    if (n < pair_ind_prob.size()-1) ofs << ", ";
                }
                ofs << endl;
            }
            ofs.close();
            cerr << "done, time spent: " << float(clock() - t_b4_write)/CLOCKS_PER_SEC << " secs" << endl << endl;
        } 
        if (f_topicword2 != "")
        {
            cerr << "writing to topicword file with vocab: " << f_topicword2 << endl;
            clock_t t_b4_write = clock();
            ofstream ofs;
            ofs.open(f_topicword2.c_str());
            for (int c=0; c<mat_posterior_prob_phi.cols(); c++)
            {
                // create vector for each column in matrix
                vector< pair<int, double> > pair_ind_prob;
                for (int r=0; r<mat_posterior_prob_phi.rows(); r++)
                {
                    pair_ind_prob.push_back( pair<int, double> (r, mat_posterior_prob_phi(r, c)) );
                }
                // sort vector by descending order, keeping track of indices
                sort( pair_ind_prob.begin() , pair_ind_prob.end() , pairCmp );

                // output top 100 word types with probability for each topic
                int numTopWords = (pair_ind_prob.size() < 100 ? pair_ind_prob.size() : 100);
                for (int n=0; n< numTopWords ; n++) 
                {
                    ofs << vocabs[ pair_ind_prob[n].first ] << ":" << pair_ind_prob[n].second;
                    if (n < numTopWords-1) ofs << ", ";
                }
                ofs << endl;
            }
            ofs.close();
            cerr << "done, time spent: " << float(clock() - t_b4_write)/CLOCKS_PER_SEC << " secs" << endl << endl;
        } 

        // debug similarity testing: sanity check to see if each training doc predicts itself as the most similar doc
        // print to stdout, convert internal 0-based doc ind to 1-based
        if ( predictSimilar && debug )
        {
            for (int j=0; j<D; j++)
            {
                clock_t t_b4_test = clock();
                int best_doc_ind = 0;
                // default to l2 distance, much faster than condprob
                if (similarity == "l2")
                    best_doc_ind = get_similar_by_l2_distance(documents[j], D, W, K, ALPHA, ETA, BURNIN_PER_DOC, S2, TAO2, KAPPA2, mat_N_phi, mat_posterior_prob_theta);
                else if (similarity == "condprob")
                    best_doc_ind = get_similar_by_condprob(documents[j], mat_posterior_prob_phi, mat_posterior_prob_theta, D);
                cerr << "testdoc " << j+1 << " similar " << best_doc_ind+1 << " time spent: " << float(clock() - t_b4_test)/CLOCKS_PER_SEC << " secs" << endl; 
            }
        }

        // similarity testing
        // print to stdout, convert internal 0-based doc ind to 1-based
        if (predictSimilar && ! debug) cerr << "\nReading input from STDIN" << endl;
        while ( predictSimilar && (! debug) && cin) 
        {
            string testdoc_line;
            getline(cin, testdoc_line);
            if (!cin.eof())
            {
                cerr << "testdoc_line " << testdoc_line << endl;
                Document * testdoc = get_str2doc(testdoc_line);
                int best_doc_ind = get_similar_by_l2_distance(testdoc, D, W, K, ALPHA, ETA, BURNIN_PER_DOC, S2, TAO2, KAPPA2, mat_N_phi, mat_posterior_prob_theta);
                cout << "most similar to doc:" << best_doc_ind+1 << endl << endl;
            }
        }

    } catch (ArgException &e)  // catch any exceptions
    { cerr << "error: " << e.error() << " for arg " << e.argId() << endl; }

}
