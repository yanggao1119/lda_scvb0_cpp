#!/bin/bash

#NOTE: uncomment to each demo
# then run:      
# $sh run_examples.sh 

mkdir -p exp

# 1. single thread, topic training with uci nips data, write result to files; runtime output such as parameter summary, perlexity is output in STDERR, redirected to log file
export OMP_NUM_THREADS=1; ./lda_scvb0 -d data_uci/docword.nips.txt -v data_uci/vocab.nips.txt -s 10 -k 100 --doctopicfile exp/doctopic.txt --topicwordfile exp/topics.txt --topicwordfile2 exp/topics.vocab.txt 

# 2. as 1., yet using four threads. Note that reported time is cpu time, therefore divide by 2-3.5 to have better idea
#export OMP_NUM_THREADS=4; ./lda_scvb0 -d data_uci/docword.nips.txt -v data_uci/vocab.nips.txt -s 10 -k 100 --doctopicfile exp/doctopic.txt --topicwordfile exp/topics.txt --topicwordfile2 exp/topics.vocab.txt 

# 3. as 2., yet running prediction mode after training, you can type one-line compact docword format, such as "1 2 3 4" and hit enter. By default use l2 distance as similarity metric and output top 50
#export OMP_NUM_THREADS=4; ./lda_scvb0 -p -d data_uci/docword.nips.txt -v data_uci/vocab.nips.txt -s 10 -k 100 --doctopicfile exp/doctopic.txt --topicwordfile exp/topics.txt --topicwordfile2 exp/topics.vocab.txt

# 4. as 3., yet running debug mode during prediction to see if each training document predicts itself as the most similar doc
#export OMP_NUM_THREADS=4; ./lda_scvb0 --debug -p -d data_uci/docword.nips.txt -v data_uci/vocab.nips.txt -s 10 -k 100 --doctopicfile exp/doctopic.txt --topicwordfile exp/topics.txt --topicwordfile2 exp/topics.vocab.txt 

# 5. as 3., yet using files at both ends of the pipe, i.e., query_input >> STDIN and STDOUT >> query_out
#export OMP_NUM_THREADS=4; cat data_uci/docword.nips.txt | python get_docword2oneline.py | ./lda_scvb0 -p -d data_uci/docword.nips.txt -v data_uci/vocab.nips.txt -s 10 -k 100 --doctopicfile exp/doctopic.txt --topicwordfile exp/topics.txt --topicwordfile2 exp/topics.vocab.txt > exp/query.out 

# 6. as 5., yet using arg switch to provide test file, this is robust to blank line; also outputting mat params from training to exp dir s.t. we don't need to go through training again but can directly load param from file
#export OMP_NUM_THREADS=4; ./lda_scvb0 -p -d data_uci/docword.nips.txt -t data_uci/docword.nips.txt -v data_uci/vocab.nips.txt --outmatdir exp -s 50 -k 100 --doctopicfile exp/doctopic.txt --topicwordfile exp/topics.txt --topicwordfile2 exp/topics.vocab.txt > exp/query.out2 

# 7. as 6., yet running prediction with mat params saved from 6
#export OMP_NUM_THREADS=4; ./lda_scvb0 -p -d data_uci/docword.nips.txt -t data_uci/docword.nips.txt -v data_uci/vocab.nips.txt --inmatdir exp -s 50 -k 100 --doctopicfile exp/doctopic.txt --topicwordfile exp/topics.txt --topicwordfile2 exp/topics.vocab.txt > exp/query.out2 

