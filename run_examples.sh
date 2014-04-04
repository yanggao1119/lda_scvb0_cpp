#!/bin/bash

#NOTE: uncomment to each demo
# then run:      
# $sh run_examples.sh 

# 1. single thread, topic training with uci nips data, write result to files; runtime output such as parameter summary, perlexity is output in STDERR, redirected to log file
export OMP_NUM_THREADS=1; ./lda_scvb0 -d data_uci/docword.nips.txt -v data_uci/vocab.nips.txt -s 10 -k 100 --doctopicfile doctopic.txt --topicwordfile topics.txt --topicwordfile2 topics.vocab.txt 2>log

# 2. as 1., yet using four threads. Note that reported time is cpu time, therefore divide by 2-3.5 to have better idea
#export OMP_NUM_THREADS=4; ./lda_scvb0 -d data_uci/docword.nips.txt -v data_uci/vocab.nips.txt -s 10 -k 100 --doctopicfile doctopic.txt --topicwordfile topics.txt --topicwordfile2 topics.vocab.txt 2>log

# 3. as 2., yet running prediction mode after training, you can type one-line compact docword format, such as "1 2 3 4" and hit enter
#export OMP_NUM_THREADS=4; ./lda_scvb0 -p -d data_uci/docword.nips.txt -v data_uci/vocab.nips.txt -s 10 -k 100 --doctopicfile doctopic.txt --topicwordfile topics.txt --topicwordfile2 topics.vocab.txt

# 4. as 3., yet running debug mode during prediction to see if each training document predicts itself as the most similar doc
#export OMP_NUM_THREADS=4; ./lda_scvb0 --debug -p -d data_uci/docword.nips.txt -v data_uci/vocab.nips.txt -s 10 -k 100 --doctopicfile doctopic.txt --topicwordfile topics.txt --topicwordfile2 topics.vocab.txt 2>log

# 5. as 3., yet using files at both ends of the pipe, i.e., query_input >> STDIN and STDOUT >> query_out
#export OMP_NUM_THREADS=4; cat data_uci/docword.nips.txt | python get_docword2oneline.py | ./lda_scvb0 -p -d data_uci/docword.nips.txt -v data_uci/vocab.nips.txt -s 10 -k 100 --doctopicfile doctopic.txt --topicwordfile topics.txt --topicwordfile2 topics.vocab.txt > query.out 2>log
