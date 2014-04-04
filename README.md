lda_scvb0_cpp
=============

Yang Gao's implementation of Jimmy Fould's SCVB0 algorithm, coded in C++ with the Eigen matrix package, burn-in for each doc is modified after Michael Hankin's vectorized method.

It does multi-threaded topic model training (simple multi-threading with the openmp package) and output training log (parameters, perplexity etc) to STDERR. After that, an optional prediction mode can read query document from STDIN and output to STDOUT the most similar document from the training set. See below for details.

usage
=====

- input follows the UCI sparse bag-of-words format, which consists of two files: 
    - a vocab file specified by -v switch, listing all unique word types one at a line; 
    - a docword file specified by -d switch, where each line is a (doc id, word id, word count) triple delimited by space. Both doc id and word id are 1-based. Word id refers to the corresponding line in the vocab file;

- output files can be specified with these switches:

   - --doctopicfile <string>
     Output result of scvb0 training, where each line represents the distribution of all topics for a document, separated by commas;
   - --topicwordfile <string>
     Output result of scvb0 training, where each line lists for a topic all word ids sorted by their probabilities in a descending order. If there are 1000 word types (i.e., 1000 lines in the vocab file), this line will have 1001 entries in the format 'wordID:weight, wordID:weight, ...wordID:weight', where wordID=0 is reserved for unknown words <unk> which may appear in query document;
   - --topicwordfile2 <string>
     Same as --topicwordfile, yet for better human consumption only lists the top 100 words and replaces word id by word type;

- when the -p switch is on, the code runs in prediction mode, i.e., after training topic model, it accepts a one-line compact docword representation from STDIN and outputs to STDOUT the id of the most similar document (1-based) from the training set. Note that:

    - format of compact docword representation: line "1 2 3 4" means that in the query document, word id 1 occurs 2 times and word id 3 occurs 4 times. Unknown words from query are all mapped to "<unk>" with word id 0.
    - similarity is by default measured as the L2 distance of topic distribution between the query and training docs, i.e. by default "--similarity l2". Another similarity measure, P(query_doc|training_doc) requires more computation and can be turned on by specifying "--similarity condprob".

- for other parameters and options, type "./lda_scvb0 --help"

-  see "run_examples.sh" for example usage.

dependencies
============
1. make sure g++ is 4.4.7 or above, since I only tested on 4.4.7 and 4.7
2. external libraries, such as Eigen and tclap, are included; therefore the code is ready to run

compiling
=========
1. in my Makefile, I have CXX=/usr/local/bin/g++-4.7, you may need to change it to your g++ path;
2. for initial build, type "make";
3. if you modify code, type "make rebuild"

todos
=====
1. multi-threading is naive and has to be speeded up with better accuracy.
2. needs profiling to speed up code. I guess Eigen's matrix operations could be speeded up by avoiding temporary copies.
3. similarity testing by kl divergence.

questions
=========
for questions, comments or to report bugs, contact: yanggao1119@gmail.com
