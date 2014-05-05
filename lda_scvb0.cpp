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
//NOTE: for creating dirs, maybe useful for sth else
#include <sys/types.h>
#include <sys/stat.h>

using namespace std;
using namespace Eigen;
using namespace TCLAP;


// check file existence
inline bool isFileExist (const std::string& name) {
    ifstream f(name.c_str());
    if (f.good()) {
        f.close();
        return true;
    } else {
        f.close();
        return false;
    }   
}


// pair comparer copied from shaofeng mo
bool pairCmpDescend( const pair<int,double>& p1, const pair<int,double>& p2){
  if( p1.second - p2.second > 0.00000000001 ){
    return true;
  }else{
    return false;
  }
}


bool pairCmpAscend( const pair<int,double>& p1, const pair<int,double>& p2){
  if( p1.second - p2.second < 0.00000000001 ){
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
        //NOTE: convert 1-based word and doc index to be internally 0-based
        w_i -= 1;
        doc->word_ind.push_back(w_i);
        doc->word_count.push_back(w_c);
    }
    /*for (int i=0; i<doc->word_ind.size(); i++)
    {
        cerr << "w_i:" << doc->word_ind[i] << "\tw_c:" << doc->word_count[i] << endl;
        
    }*/
    return doc;
}


void read_mat(const string & filename, 
                MatrixXd & mat)
{
    cerr << "Start reading mat file " << filename << endl;
    clock_t t_b4_read = clock();

    // get dimension from matrix file, col element delimited by space
    ifstream f(filename.c_str());
    string line;
    int rowDim=0, colDim=0;
    for(; getline(f, line); rowDim++) 
    {
        if (rowDim==0)
        {
            std::istringstream iss(line);
            string tok;
            while ( getline(iss, tok, ' ') ) colDim++;
        }
    }

    // populate matrix
    ifstream ff(filename.c_str());
    mat = MatrixXd::Constant(rowDim, colDim, 0);
    //TODO: why this initialization does not work?
    //mat(rowDim, colDim);
    for (int r=0; getline(ff, line); r++)
    {
        std::istringstream iss2(line);
        int c=0;
        string tok;
        while ( getline( iss2, tok, ' ' ) ) {
            //TODO: atof may lost precision, if so need to convert to double
            mat(r, c) = atof(tok.c_str());
            c++;
        }
    }

    cerr << "dimension: " << rowDim << " by " << colDim << endl;
    cerr << "done reading mat file, time spent: " << float(clock() - t_b4_read)/CLOCKS_PER_SEC << " secs" << endl << endl;
}


void read_vocab(string file, vector<string> & vocabs) 
{
    cerr << "Start reading vocab file " << file << endl;
    clock_t t_b4_read = clock();

    ifstream ifs(file.c_str());
    for (string line ; getline(ifs, line); )
    {
       vocabs.push_back(line); 
    }
    cerr << "done reading vocab file, time spent: " << float(clock() - t_b4_read)/CLOCKS_PER_SEC << " secs" << endl << endl;
}


Document** read_docword(string file, int & D, int & W, int & C) 
{
    cerr << "Start reading docword file " << file << endl;
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
            W = atoi(line.c_str());
        }
        else if (line_count != 3) 
        {
			//NOTE: istringstream is ambiguous with tclap; compiling error w/o std scoping
            std::istringstream iss(line);
            int d_j, w_i, w_c;
            iss >> d_j >> w_i >> w_c;
            C += w_c;
            //NOTE: convert 1-based word and doc index to be internally 0-based
            w_i -= 1;
            d_j -= 1;
            //cerr << d_j << " " << w_i << " " << w_c << endl;
            documents[d_j]->word_ind.push_back(w_i);
            documents[d_j]->word_count.push_back(w_c);
        }
    }
    cerr << "D:" << D << "\tW:" << W << "\tC:" << C << endl; 
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

    // empty doc, return w/o burnin
    if (C_j == 0)   
    {
        return mat_N_theta_j;
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


void get_similar_by_skl(vector< pair<int, double> > & pair_docind_score,
                                const Document * testdoc, 
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
    // given testdoc, rank training doc by symmetric kl divergence: smaller divergence, higher rank
    MatrixXd mat_N_theta_test = ((MatrixXd::Random(1, K)*0.5).array() + 1).matrix();
    mat_N_theta_test = burnin_doc_j(testdoc, W, K, ALPHA, ETA, BURNIN_PER_DOC, S2, TAO2, KAPPA2, mat_N_phi, mat_N_theta_test);
    MatrixXd mat_posterior_prob_theta_test = MatrixXd::Constant(1, K, 0);
    get_mat_posterior_prob_theta(mat_N_theta_test, ALPHA, mat_posterior_prob_theta_test);
  
    for (int j=0; j<D; j++)
    {
        double dist = 0;
        for (int c=0; c<mat_posterior_prob_theta_test.cols(); c++)
        {
            dist += mat_posterior_prob_theta_test(0, c) * log2( mat_posterior_prob_theta_test(0, c) / mat_posterior_prob_theta(j, c) ); 
        }
        for (int c=0; c<mat_posterior_prob_theta.cols(); c++)
        {
            dist += mat_posterior_prob_theta(j, c) * log2( mat_posterior_prob_theta(j, c) / mat_posterior_prob_theta_test(0, c) ); 
        }
        dist /= 2;
        pair_docind_score.push_back( pair<int, double> (j, dist) );
    }
    sort( pair_docind_score.begin() , pair_docind_score.end() , pairCmpAscend );
}


void update_streaming_naive(
                                const Document * streamdoc, 
                                const int & D,
                                const int & W,
                                const int & C,
                                const int & K,
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
                                const MatrixXd & mat_N_theta,
                                MatrixXd & mat_N_phi,
                                MatrixXd & mat_N_phi_hat,
                                MatrixXd & mat_N_Z_hat,
                                MatrixXd & mat_N_Z, 
                                unsigned long & doc_iter)
{
    doc_iter ++;

    // naive streaming udpate: only mat_N_phi and mat_posterior_prob_phi
    MatrixXd mat_N_theta_test = ((MatrixXd::Random(1, K)*0.5).array() + 1).matrix();
    mat_N_theta_test = burnin_doc_j(streamdoc, W, K, ALPHA, ETA, BURNIN_PER_DOC, S2, TAO2, KAPPA2, mat_N_phi, mat_N_theta_test);
 
    double minibatches_per_corpus = (1.0 * C) / M; 
                
    // final update for doc
    for (int i=0; i<streamdoc->word_ind.size(); i++)
    {
        int w_i = streamdoc->word_ind[i];
        int w_c = streamdoc->word_count[i];
        // eqn 5
        MatrixXd mat_gamma_i_j = (((mat_N_phi.row(w_i).array() + ETA)*(mat_N_theta_test.row(0).array() + ALPHA))/( mat_N_Z.array() + ETA*W)).matrix();
        mat_gamma_i_j /= mat_gamma_i_j.sum(); 
        
        // update hat            
        mat_N_phi_hat.row(w_i) += minibatches_per_corpus*w_c*mat_gamma_i_j;
        mat_N_Z_hat += minibatches_per_corpus*w_c*mat_gamma_i_j;
    }
                 
    if (doc_iter % M == 0)
    {
        // batch update
        double rho_phi = S/pow(TAO+doc_iter, KAPPA);
        // eqn 7
        mat_N_phi = (1-rho_phi)*mat_N_phi + rho_phi*mat_N_phi_hat;
        // eqn 8
        mat_N_Z = (1-rho_phi)*mat_N_Z + rho_phi*mat_N_Z_hat;

        mat_N_phi_hat *= 0;
        mat_N_Z_hat *= 0;
        mat_N_Z = mat_N_phi.colwise().sum();

    } // finish minibatch update

}


void get_similar_by_l2_distance(vector< pair<int, double> > & pair_docind_score,
                                const Document * testdoc, 
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
    // given testdoc, rank training doc by similarity: smaller l2 distance, higher rank
    // l2 distance is less expensive to compute than condprob
    MatrixXd mat_N_theta_test = ((MatrixXd::Random(1, K)*0.5).array() + 1).matrix();
    mat_N_theta_test = burnin_doc_j(testdoc, W, K, ALPHA, ETA, BURNIN_PER_DOC, S2, TAO2, KAPPA2, mat_N_phi, mat_N_theta_test);
    MatrixXd mat_posterior_prob_theta_test = MatrixXd::Constant(1, K, 0);
    get_mat_posterior_prob_theta(mat_N_theta_test, ALPHA, mat_posterior_prob_theta_test);
  
    for (int j=0; j<D; j++)
    {
        double dist = (mat_posterior_prob_theta_test - mat_posterior_prob_theta.row(j)).norm();
        pair_docind_score.push_back( pair<int, double> (j, dist) );
    }

    sort( pair_docind_score.begin() , pair_docind_score.end() , pairCmpAscend );
}


void get_similar_by_logcondprob(vector< pair<int, double> > & pair_docind_score,
                            const Document * testdoc, 
                            const MatrixXd & mat_posterior_prob_phi,
                            const MatrixXd & mat_posterior_prob_theta,
                            const int & D)
{
    // given testdoc, rank training doc by log of conditional prob p(q|d): bigger p, higher rank
    // p(q|d) is from eqn 9 in "Probabilistic Topic Models" by Steyvers and Griffiths
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
        pair_docind_score.push_back( pair<int, double> (j, log_doc_prob) );
    }

    sort( pair_docind_score.begin() , pair_docind_score.end() , pairCmpDescend );
}


void scvb0(Document ** documents, 
          const int & D,
          const int & W,
          const int & C,
          const int & K,
          const int & SWEEP,
          const double & STOP_CHANGE,
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
          MatrixXd & mat_posterior_prob_theta,
          unsigned long & doc_iter)
{
    cerr << "Start scvb0" << endl;
    clock_t t_b4_scvb0 = clock();

    //randomization is key, yet how to randomize and whether or not to normalize is not important 
    cerr << "Initializing..." << endl;
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
    double prev_perplexity = 0.0;

    clock_t t_b4_all_sweep = clock();
    for (int s=0; s<SWEEP; s++)
    {
        clock_t t_b4_this_sweep = clock();
        #pragma omp parallel for
        for (int m=0; m<number_minibatches; m++)
        {
            MatrixXd mat_N_phi_hat = MatrixXd::Constant(W, K, 0);
            MatrixXd mat_N_Z_hat = MatrixXd::Constant(1, K, 0);
            MatrixXd mat_N_Z = mat_N_phi.colwise().sum();

            int minibatch_start = m*M;
            int minibatch_end = ( (m+1)*M > D ? D : (m+1)*M );

            //NOTE: document iteration accumulates from sweeps
            //cerr << "batch " << m << " start " << minibatch_start << " end " <<  minibatch_end << endl;
            for (int j=minibatch_start; j<minibatch_end; j++)
            {
                //for large dataset, give poor audience some clue where we are; multiple reports in multi-threaded mode
                if ((j+1)%10000==0) cerr << "processing doc " << (j+1) << endl;
 
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
                clock_t t_after_doc_iter = clock();
                if (doc_iter % REPORT_PERPLEXITY_ITER == 0)
                {
                    get_mat_posterior_prob_phi(mat_N_phi, ETA, mat_posterior_prob_phi);
                    get_mat_posterior_prob_theta(mat_N_theta, ALPHA, mat_posterior_prob_theta);
                    double perplexity = get_perplexity(documents, D, C, mat_posterior_prob_phi, mat_posterior_prob_theta);

                    cerr << "done doc_iter " << doc_iter << ", time spent: " << float(t_after_doc_iter - t_b4_all_sweep)/CLOCKS_PER_SEC << " secs, perplexity: " << perplexity << endl;
                }*/

            } // finish all doc updates in batch

            // batch update
            double rho_phi = S/pow(TAO+doc_iter, KAPPA);
            // eqn 7
            mat_N_phi = (1-rho_phi)*mat_N_phi + rho_phi*mat_N_phi_hat;
            // eqn 8
            //TODO: this update seems useless, check and remove?
            mat_N_Z = (1-rho_phi)*mat_N_Z + rho_phi*mat_N_Z_hat;
        } // finish minibatch update
        
        // report perplexity after finishing a sweep
        get_mat_posterior_prob_phi(mat_N_phi, ETA, mat_posterior_prob_phi);
        get_mat_posterior_prob_theta(mat_N_theta, ALPHA, mat_posterior_prob_theta);
        double perplexity = get_perplexity(documents, D, C, mat_posterior_prob_phi, mat_posterior_prob_theta);
       
        clock_t t_after_this_sweep = clock(); 
        cerr << "done sweep " << s << ", time spent: " << float(t_after_this_sweep - t_b4_this_sweep)/CLOCKS_PER_SEC << " secs, perplexity: " << perplexity << endl;

        // check convergence
        if (s > 0)
        {
            double relative_pp_change = (prev_perplexity - perplexity) / prev_perplexity;
            if (relative_pp_change <= STOP_CHANGE)
            {
                cerr << "relative perplexity change <= " << STOP_CHANGE << " , stop scvb0" << endl;
                break;            
            }
        }
        prev_perplexity = perplexity;        

        // check timeout!
        if ( float(t_after_this_sweep - t_b4_scvb0)/CLOCKS_PER_SEC > TIME_LIMIT) 
        {
            cerr << "reaching time limit " << TIME_LIMIT << " secs, stop scvb0" << endl;
            break;
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
        //TODO: change name to docwordTrainInFileArg
        ValueArg<string> docwordInFileArg("d","docwordfile","Path to docword file for training topic model, in uci sparse bag-of-words format, where each line is a (doc id, word id, word count) triple delimited by space. Both doc id and word id are 1-based. Word id refers to the corresponding line in the vocab file", true, "","string");
        cmd.add( docwordInFileArg );

        ValueArg<string> vocabInFileArg("v","vocabfile","Path to vocab file associated with docword file for training topic model, in uci sparse bag-of-words format", true, "","string");
        cmd.add( vocabInFileArg );

        ValueArg<string> docwordTestInFileArg("t","docwordtestfile","Path to docword file for similarity testing, in uci sparse bag-of-words format. Output similarity prediction for each file to STDOUT. Note that test file can also be piped from STDIN in a compact oneline format, yet this arg switch is preferred to STDIN when there are empty lines", false, "","string");
        cmd.add( docwordTestInFileArg );

        ValueArg<string> matParamInDirArg("","inmatdir", "directory of previously trained model to be loaded, which contain four files: {mat_N_phi, mat_N_theta, mat_posterior_prob_phi, mat_posterior_prob_theta}.txt", false, "","string");
        cmd.add( matParamInDirArg );

        ValueArg<string> matParamOutDirArg("","outmatdir", "directory to dump trained model, which contain four files: {mat_N_phi, mat_N_theta, mat_posterior_prob_phi, mat_posterior_prob_theta}.txt", false, "","string");
        cmd.add( matParamOutDirArg );

        ValueArg<string> doctopicOutFileArg("","doctopicfile","Output result of scvb0 training, where each line represents the distribution of all topics for a document, separated by commas", false, "","string");
        cmd.add( doctopicOutFileArg );

        ValueArg<string> topicwordOutFileArg("","topicwordfile","Output result of scvb0 training, where each line lists for a topic all word ids sorted by their probabilities in a descending order. If there are 1000 word types (i.e., 1000 lines in the vocab file), this line will have 1000 entries in the format 'wordID:weight, wordID:weight, ...wordID:weight'", false, "","string");
        cmd.add( topicwordOutFileArg );

        ValueArg<string> topicwordOutFile2Arg("","topicwordfile2","Same as --topicwordfile, yet for better human consumption only lists the top 100 words and replaces word id by word type", false, "","string");
        cmd.add( topicwordOutFile2Arg );

        ValueArg<int> sweepArg("s","sweep","Number of sweeps over docword file", false, 10,"int");
        cmd.add( sweepArg );

        ValueArg<double> stopChangeArg("","stopchange","Optional convergence criterion: stop scvb0 training if the relative change of perplexity is below this number. This is calculated at the end of a sweep, as (prev_pp - curr_pp)/prev_pp", false, 0.0005,"double");
        cmd.add( stopChangeArg );

        ValueArg<double> iterDiscountArg("","iterdiscount","Discount iteration starting number for streaming mode", false, 1.0,"double");
        cmd.add( iterDiscountArg );



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

        ValueArg<int> timeLimitArg("","timelimit","Time limit for training, in seconds", false, 360000,"int");
        cmd.add( timeLimitArg );

        ValueArg<int> reportPerpIterArg("","reportperp","Report training perplexity for how many document iterations", false, 10000,"int");
        cmd.add( reportPerpIterArg );

        ValueArg<string> similarMetricArg("","similarmetric","Method to measure similarity between the query and training docs, default to L2 distance of topic distribution. Another similarity measure, P(query_doc|training_doc) requires more computation and can be turned on by specifying condprob as value, options: l2 for l2 distance over topic distribution; skl for symmetric kl divergence over topic distribution; condprob for p(query|doc)", false, "l2","string");
        cmd.add( similarMetricArg );

        ValueArg<int> similarSizeArg("","similarsize","for similarity test, how many top similar docs to report", false, 50,"int");
        cmd.add( similarSizeArg );

        SwitchArg predictSimilarSwitch("p","predict","optional prediction mode, i.e., after training topic model, it accepts a one-line compact docword representation from STDIN and outputs to STDOUT the id of the most similar document (1-based) from the training set", false);
        cmd.add( predictSimilarSwitch );

        //TODO: advanced streaming that update vocab also
        SwitchArg naiveStreamingSwitch("","naivestreaming","running a naive streaming mode, i.e., updating the phi matrix as we go through each testdoc", false);
        cmd.add( naiveStreamingSwitch );

        //TODO: function to specify random seed explicitly
        SwitchArg debugSwitch("","debug","running debug mode, specifying random seed and sanity check to see if each training doc predicts itself as the most similar doc", false);
        cmd.add( debugSwitch );

        cmd.parse( argc, argv );

        // Get the value parsed by each arg. 
        const string f_docword = docwordInFileArg.getValue();
        const string f_vocab = vocabInFileArg.getValue();
        const string f_docword_test = docwordTestInFileArg.getValue();
        const string d_inmat = matParamInDirArg.getValue();
        const string d_outmat = matParamOutDirArg.getValue();
        const string f_doctopic = doctopicOutFileArg.getValue();
        const string f_topicword = topicwordOutFileArg.getValue();
        const string f_topicword2 = topicwordOutFile2Arg.getValue();

        const int SWEEP = sweepArg.getValue();
        const double STOP_CHANGE = stopChangeArg.getValue();
        const double ITER_DISCOUNT = iterDiscountArg.getValue();
        int K = numTopicArg.getValue();

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

        const string SIMILAR_METRIC = similarMetricArg.getValue();
        const int SIMILAR_SIZE = similarSizeArg.getValue();

        const bool predictSimilar = predictSimilarSwitch.getValue();
        const bool naiveStreaming = naiveStreamingSwitch.getValue();
        const bool debug = debugSwitch.getValue();

        // report parameters
        cerr << "Parameters:\n";

        cerr << "Input train docword file: " << f_docword << endl;
        cerr << "Input train vocab file: " << f_vocab << endl;
        cerr << "Input test docword file: " << f_docword_test << endl;
        cerr << "Input matrix param dir: " << d_inmat << endl;
        cerr << "Output matrix param dir: " << d_outmat << endl;
        cerr << "Output doctopic file: " << f_doctopic << endl;
        cerr << "Output topicword file: " << f_topicword << endl;
        cerr << "Output topicword2 file: " << f_topicword2 << endl;

        cerr << "K: " << K << "\tSWEEP: " << SWEEP << "\tSTOP_CHANGE: " << STOP_CHANGE << endl;
        cerr << "ITER_DISCOUNT: " << ITER_DISCOUNT << endl;
        cerr << "ALPHA: " << ALPHA << "\tETA: " << ETA << endl;
        cerr << "M: " << M << "\tBURNIN_PER_DOC: " << BURNIN_PER_DOC << endl;
        cerr << "S: " << S << "\tTAO: " << TAO << "\tKAPPA: " << KAPPA << endl; 	
        cerr << "S2: " << S2 << "\tTAO2: " << TAO2 << "\tKAPPA2: " << KAPPA2 << endl; 	
        cerr << "RANDSEED_phi: " << RANDSEED_phi << "\tRANDSEED_theta: " << RANDSEED_theta << endl;
        cerr << "TIME_LIMIT: " << TIME_LIMIT << "\tREPORT_PERPLEXITY_ITER: " << REPORT_PERPLEXITY_ITER << endl;
        cerr << "SIMILAR_METRIC: " << SIMILAR_METRIC << "\tSIMILAR_SIZE: " << SIMILAR_SIZE << endl;
        
        cerr << "Switches:\n";
        cerr << "predictSimilar: " << (predictSimilar ? "true" : "false") << endl;
        cerr << "naiveStreaming: " << (naiveStreaming ? "true" : "false") << endl;
        cerr << "debug: " << (debug ? "true" : "false") << endl;
        cerr << endl;

        // read vocab file
        vector<string> vocabs;
        read_vocab(f_vocab, vocabs);
        //for (int i=0; i<vocabs.size(); i++) cerr << vocabs[i] << endl;
        //cerr << "total vocab " << vocabs.size() << endl;

        // read docword file
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

        // model option 1: directly load matrix params from files
        // model option 2: run scvb0 training
        MatrixXd mat_N_phi, mat_N_theta, mat_posterior_prob_phi, mat_posterior_prob_theta;

        string f_in_mat_N_phi                = d_inmat + "/" + "mat_N_phi.txt";
        string f_in_mat_N_theta              = d_inmat + "/" "mat_N_theta.txt";
        string f_in_mat_posterior_prob_phi   = d_inmat + "/" + "mat_posterior_prob_phi.txt";
        string f_in_mat_posterior_prob_theta = d_inmat + "/" + "mat_posterior_prob_theta.txt";

        unsigned long doc_iter = 0;

        if (isFileExist(f_in_mat_N_phi) && 
            isFileExist(f_in_mat_N_theta) && 
            isFileExist(f_in_mat_posterior_prob_phi) && 
            isFileExist(f_in_mat_posterior_prob_theta))
        {
            cerr << "Loading matrix params from files" << endl << endl;
            read_mat(f_in_mat_N_phi, mat_N_phi);
            W = mat_N_phi.rows();
            K = mat_N_phi.cols();
            //TODO: assert that K is same as provided by switch
            read_mat(f_in_mat_N_theta, mat_N_theta);
            D = mat_N_theta.rows();
            read_mat(f_in_mat_posterior_prob_phi, mat_posterior_prob_phi);
            read_mat(f_in_mat_posterior_prob_theta, mat_posterior_prob_theta);
            //TODO: this is actually a cheap bug, need to export sweeps of prev run
            doc_iter = D;
        }
        else
        {
            // scvb0
            scvb0(documents, 
                  D,
                  W,
                  C,
                  K,
                  SWEEP,
                  STOP_CHANGE,
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
                  mat_posterior_prob_theta,
                  doc_iter);
        }

        // streaming mode
        if (naiveStreaming)
        {
            doc_iter *= ITER_DISCOUNT;

            MatrixXd mat_N_phi_hat = MatrixXd::Constant(W, K, 0);
            MatrixXd mat_N_Z_hat = MatrixXd::Constant(1, K, 0);
            MatrixXd mat_N_Z = mat_N_phi.colwise().sum();

            cerr << "\nReading streaming doc from STDIN" << endl;
            unsigned long counter = 0;
            while (cin) 
            {
                string streamdoc_line;
                getline(cin, streamdoc_line);
                if (!cin.eof())
                {
                    counter ++;
                    clock_t t_b4_test = clock();
                    Document * streamdoc = get_str2doc(streamdoc_line);

                    update_streaming_naive(
                                            streamdoc, 
                                            D,
                                            W,
                                            C,
                                            K,
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
                                            mat_N_theta,
                                            mat_N_phi,
                                            mat_N_phi_hat,
                                            mat_N_Z_hat,
                                            mat_N_Z,
                                            doc_iter);
                }
            }
            cerr << "Streamed doc#: " << counter << endl;
            get_mat_posterior_prob_phi(mat_N_phi, ETA, mat_posterior_prob_phi);
        }

        // output results to files

        // output matrix params s.t. we don't need to train lda again
        if (d_outmat != "")
        {
            cerr << "Saving matrix params of lda training to directory: " << d_outmat << endl;
            //NOTE: modified from http://pubs.opengroup.org/onlinepubs/009695399/functions/mkdir.html
            //create dir with read/write/search permissions for owner and group, and with read/search permissions  
            //return 0 if created; -1 if cannot create or dir already exists
            //TODO: check whether cannot create or dir already exists when returning -1
            //TODO: check whether this works with absoluate path
            int mkdirStat = mkdir(d_outmat.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);

            string f_out_mat_N_phi                = d_outmat + "/" + "mat_N_phi.txt";
            string f_out_mat_N_theta              = d_outmat + "/" "mat_N_theta.txt";
            string f_out_mat_posterior_prob_phi   = d_outmat + "/" + "mat_posterior_prob_phi.txt";
            string f_out_mat_posterior_prob_theta = d_outmat + "/" + "mat_posterior_prob_theta.txt";

            vector< pair<string, MatrixXd *> > outPairs;
            outPairs.push_back( pair<string, MatrixXd *> (f_out_mat_N_phi, &mat_N_phi) );
            outPairs.push_back( pair<string, MatrixXd *> (f_out_mat_N_theta, &mat_N_theta) );
            outPairs.push_back( pair<string, MatrixXd *> (f_out_mat_posterior_prob_phi, &mat_posterior_prob_phi) );
            outPairs.push_back( pair<string, MatrixXd *> (f_out_mat_posterior_prob_theta, &mat_posterior_prob_theta) );

            for (int p=0; p<outPairs.size(); p++)
            {
                cerr << "Writing to " << outPairs[p].first << endl;
                clock_t t_b4_write = clock();
                IOFormat matrixFormat(FullPrecision, DontAlignCols, " ", "\n", "", "", "", "");
                ofstream ofs;
                ofs.open(outPairs[p].first.c_str());
                ofs << outPairs[p].second->format(matrixFormat) << endl;
                ofs.close();
                cerr << "done, time spent: " << float(clock() - t_b4_write)/CLOCKS_PER_SEC << " secs" << endl << endl;
            }
        }

 
        // output doc-topic, topic-word for human consumption
        if (f_doctopic != "")
        {
            cerr << "Writing to doctopic file: " << f_doctopic << endl;
            clock_t t_b4_write = clock();
            IOFormat matrixFormat(FullPrecision, DontAlignCols, ", ", "\n", "", "", "", "");
            ofstream ofs;
            ofs.open(f_doctopic.c_str());
            ofs << mat_posterior_prob_theta.format(matrixFormat) << endl;
            ofs.close();
            cerr << "done, time spent: " << float(clock() - t_b4_write)/CLOCKS_PER_SEC << " secs" << endl << endl;
        } 
        if (f_topicword != "")
        {
            cerr << "Writing to topicword file: " << f_topicword << endl;
            clock_t t_b4_write = clock();
            ofstream ofs;
            ofs.open(f_topicword.c_str());
            for (int c=0; c<mat_posterior_prob_phi.cols(); c++)
            {
                // create vector for each column in matrix
                vector< pair<int, double> > pair_wordind_prob;
                for (int r=0; r<mat_posterior_prob_phi.rows(); r++)
                {
                    pair_wordind_prob.push_back( pair<int, double> (r, mat_posterior_prob_phi(r, c)) );
                }
                // sort vector by descending order, keeping track of indices
                sort( pair_wordind_prob.begin() , pair_wordind_prob.end() , pairCmpDescend );

                // output all word ids with probability for each topic
                // convert internal 0-based word ind to 1-based
                for (int n=0; n< pair_wordind_prob.size() ; n++) 
                {
                    ofs << pair_wordind_prob[n].first + 1 << ":" << pair_wordind_prob[n].second;
                    if (n < pair_wordind_prob.size()-1) ofs << ", ";
                }
                ofs << endl;
            }
            ofs.close();
            cerr << "done, time spent: " << float(clock() - t_b4_write)/CLOCKS_PER_SEC << " secs" << endl << endl;
        } 
        if (f_topicword2 != "")
        {
            cerr << "Writing to topicword file with vocab: " << f_topicword2 << endl;
            clock_t t_b4_write = clock();
            ofstream ofs;
            ofs.open(f_topicword2.c_str());
            for (int c=0; c<mat_posterior_prob_phi.cols(); c++)
            {
                // create vector for each column in matrix
                vector< pair<int, double> > pair_wordind_prob;
                for (int r=0; r<mat_posterior_prob_phi.rows(); r++)
                {
                    pair_wordind_prob.push_back( pair<int, double> (r, mat_posterior_prob_phi(r, c)) );
                }
                // sort vector by descending order, keeping track of indices
                sort( pair_wordind_prob.begin() , pair_wordind_prob.end() , pairCmpDescend );

                // output top 100 word types with probability for each topic
                int numTopWords = (pair_wordind_prob.size() < 100 ? pair_wordind_prob.size() : 100);
                for (int n=0; n< numTopWords ; n++) 
                {
                    ofs << vocabs[ pair_wordind_prob[n].first ] << ":" << pair_wordind_prob[n].second;
                    if (n < numTopWords-1) ofs << ", ";
                }
                ofs << endl;
            }
            ofs.close();
            cerr << "done, time spent: " << float(clock() - t_b4_write)/CLOCKS_PER_SEC << " secs" << endl << endl;
        } 

        /////////////////////////////////////////////////////////////
        //the following are optional, subsequent modes after training
        /////////////////////////////////////////////////////////////
        
        // mode 1: output to STDERR perplexity of the test file
        //TODO: jimmy foulds says this test is cheating, because we optimize theta for test doc first. To not cheat, there are two ways:
        //1) split each test doc into two parts, making sure words are disjoint; then estimate theta on one part and calculate perplexity on the other
        //2) left-to-right as Wallach et al., evaluation methods for topic models
/*        if (f_docword_test != "")
        {
            cerr << "Compute perplexity of test corpus" << endl;
            clock_t t_b4_test = clock();

            int D_test=0, W_test=0, C_test=0;
            Document** documents_test = read_docword(f_docword_test, D_test, W_test, C_test); 
 
            MatrixXd mat_N_theta_test = MatrixXd::Constant(D_test, K, 0.0);
            for (int j=0; j<D_test; j++)
            {
                mat_N_theta_test.row(j) = burnin_doc_j(documents_test[j], W, K, ALPHA, ETA, BURNIN_PER_DOC, S2, TAO2, KAPPA2, mat_N_phi, mat_N_theta_test.row(j));
            }
            MatrixXd mat_posterior_prob_theta_test;
            get_mat_posterior_prob_theta(mat_N_theta_test, ALPHA, mat_posterior_prob_theta_test);
            double perplexity_test = get_perplexity(documents_test, D_test, C_test, mat_posterior_prob_phi, mat_posterior_prob_theta);
            cerr << "test perplexity: " << perplexity_test << endl; 
            cerr << "done, time spent: " << float(clock() - t_b4_test)/CLOCKS_PER_SEC << " secs" << endl << endl;
        }
*/
        // mode 2.1: similarity testing for file from arg switch
        if (predictSimilar && f_docword_test != "")
        {
            cerr << "Start similarity prediction for test corpus: " << f_docword_test << endl;

            int D_test=0, W_test=0, C_test=0;
            Document** documents_test = read_docword(f_docword_test, D_test, W_test, C_test); 

            for (int tj=0; tj<D_test; tj++)
            {

                Document * testdoc = documents_test[tj];
                
                clock_t t_b4_test = clock();
                   
                // get paired (doc_ind, score) vector with similarity measure
                vector< pair<int, double> > pair_docind_score;
                if (SIMILAR_METRIC == "l2")
                    get_similar_by_l2_distance(pair_docind_score, testdoc, D, W, K, ALPHA, ETA, BURNIN_PER_DOC, S2, TAO2, KAPPA2, mat_N_phi, mat_posterior_prob_theta);
                else if (SIMILAR_METRIC == "condprob")
                    get_similar_by_logcondprob(pair_docind_score, testdoc, mat_posterior_prob_phi, mat_posterior_prob_theta, D);
                else if (SIMILAR_METRIC == "skl")
                    get_similar_by_skl(pair_docind_score, testdoc, D, W, K, ALPHA, ETA, BURNIN_PER_DOC, S2, TAO2, KAPPA2, mat_N_phi, mat_posterior_prob_theta);
                // output to STDOUT in one line
                int numTopSimilar = (pair_docind_score.size() < SIMILAR_SIZE ? pair_docind_score.size() : SIMILAR_SIZE);
                for (int p=0; p<numTopSimilar; p++)
                {
                    cout << pair_docind_score[p].first+1 << ":" << pair_docind_score[p].second;
                    if (p < numTopSimilar-1) cout << " ";
                }
                cout << endl;
                cerr << "testdoc: " << tj+1 << " time spent: " << float(clock() - t_b4_test)/CLOCKS_PER_SEC << " secs" << endl; 
            }
        }

        // mode 2.2: similarity testing for file from STDIN, note: cannot handle blank lines!
        // input from STDIN, output to STDOUT, convert internal 0-based doc ind to 1-based
        else if (predictSimilar && ! debug)
        {
            cerr << "\nReading input from STDIN" << endl;
            long counter = 0;
            while (cin) 
            {
                string testdoc_line;
                getline(cin, testdoc_line);
                if (!cin.eof())
                {
                    counter ++;
                    clock_t t_b4_test = clock();
                    Document * testdoc = get_str2doc(testdoc_line);
    
                    // get paired (doc_ind, score) vector with similarity measure
                    vector< pair<int, double> > pair_docind_score;
                    if (SIMILAR_METRIC == "l2")
                        get_similar_by_l2_distance(pair_docind_score, testdoc, D, W, K, ALPHA, ETA, BURNIN_PER_DOC, S2, TAO2, KAPPA2, mat_N_phi, mat_posterior_prob_theta);
                    else if (SIMILAR_METRIC == "condprob")
                        get_similar_by_logcondprob(pair_docind_score, testdoc, mat_posterior_prob_phi, mat_posterior_prob_theta, D);
                    else if (SIMILAR_METRIC == "skl")
                        get_similar_by_skl(pair_docind_score, testdoc, D, W, K, ALPHA, ETA, BURNIN_PER_DOC, S2, TAO2, KAPPA2, mat_N_phi, mat_posterior_prob_theta);
    
                    // output to STDOUT in one line
                    int numTopSimilar = (pair_docind_score.size() < SIMILAR_SIZE ? pair_docind_score.size() : SIMILAR_SIZE);
                    for (int p=0; p<numTopSimilar; p++)
                    {
                        cout << pair_docind_score[p].first+1 << ":" << pair_docind_score[p].second;
                        if (p < numTopSimilar-1) cout << " ";
                    }
                    cout << endl;
                    cerr << "testdoc: " << counter << " time spent: " << float(clock() - t_b4_test)/CLOCKS_PER_SEC << " secs" << endl; 
                }
            }
        }

        // mode 3: debug training and similarity testing: sanity check to see if each training doc predicts itself as the most similar doc
        else if ( predictSimilar && debug )
        {
            cerr << "\nDebugging training and similarity test" << endl;
            for (int j=0; j<D; j++)
            {
                clock_t t_b4_test = clock();
                vector< pair<int, double> > pair_docind_score;
                if (SIMILAR_METRIC == "l2")
                    get_similar_by_l2_distance(pair_docind_score, documents[j], D, W, K, ALPHA, ETA, BURNIN_PER_DOC, S2, TAO2, KAPPA2, mat_N_phi, mat_posterior_prob_theta);
                else if (SIMILAR_METRIC == "condprob")
                    get_similar_by_logcondprob(pair_docind_score, documents[j], mat_posterior_prob_phi, mat_posterior_prob_theta, D);
                if (pair_docind_score[0].first == j)
                    cerr << "training doc " << j+1 << " ok";
                else
                    cerr << "training doc " << j+1 << " does not predict itself as most similar";
                cerr << ", time spent: " << float(clock() - t_b4_test)/CLOCKS_PER_SEC << " secs" << endl; 
            }
        }

    } catch (ArgException &e)  // catch any exceptions
    { cerr << "error: " << e.error() << " for arg " << e.argId() << endl; }

}
