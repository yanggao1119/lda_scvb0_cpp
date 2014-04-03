lda_scvb0_cpp
=============
Yang Gao's implementation of Jimmy Fould's SCVB0 algorithm, coded in C++ with Eigen matrix package, burn-in for each doc is modified after Michael Hankin's vectorized method.

usage
=====
1. input follows UCI sparse bag-of-words format, consisting of two files: 
    - a vocab file specified by -v switch, listing all unique word types one at a line 
    - a docword file specified by -d switch,  listing the (docid, wordid, count) triple, separated by space, where docid and wordid are 1-based and wordid refers to the corresponding line in the vocab file
2. output files can be specified with these switches:

   - --doctopicfile <string>
     Output result of scvb0 training, where each line represents the
     distribution of the topics for a document, separated by commas

   - --topicwordfile <string>
     Output result of scvb0 training, where each line represents the top
     100 words in the descending order of probability given the topic. Each
     line has the format: 'wordID1:weight1,
     wordID2:weight2...wordID100:weight100', for wordID refer to vocab
     file

   - --topicwordfile2 <string>
     Same as --topicwordfile, yet wordID is replaced by word type for
     better human consumption

3. when -p switch is on, the code runs in prediction mode, i.e., after training topic model, it accepts query document from STDIN and outputs to STDOUT the most similar document id from training set. Similarity measure is by default computed as L2 distance of topic distribution of the query and training docs.

4. for other parameters and options, type "./lda_scvb0 --help"

5. see "run_examples.sh" for example usage.

dependencies
============
1. make sure g++ is 4.4.7 or above, since I only tested on 4.4.7 and 4.7
2. external libraries, such as eigen and tclap, are included; therefore the code is ready to run

compiling
=========
1. for initial build, type "make";
2. if you modify code, type "make rebuild"

