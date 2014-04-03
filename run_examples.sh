#!/bin/bash

# notes:
#   make sure g++-4.4.7 or above, since I only tested on 4.4.7 and 4.7
#   run with:
#   $ sh run_lda_scvb0.sh > log

# experiments
# single thread, topic training with uci nips data, write result to files
#export OMP_NUM_THREADS=1; ./lda_scvb0 --debug -p -d data_uci/docword.nips.txt -v data_uci/vocab.nips.txt -s 10 -k 100 --doctopicfile doctopic.txt --topicwordfile topics.txt --topicwordfile2 topics.vocab.txt

# single thread, add -p to run prediction debugging to see if we predict each training doc as its most similar doc, based on l2 distance on topic distribution
#export OMP_NUM_THREADS=1; ./lda_scvb0 --debug -p -d data_uci/docword.nips.txt -v data_uci/vocab.nips.txt -s 10 -k 100 --doctopicfile doctopic.txt --topicwordfile topics.txt --topicwordfile2 topics.vocab.txt

# four threads, note that reported time is cpu time, therefore divide by 2-3.5 to have better idea
#export OMP_NUM_THREADS=4; ./lda_scvb0 --debug -p -d data_uci/docword.nips.txt -v data_uci/vocab.nips.txt -s 10 -k 100 --doctopicfile doctopic.txt --topicwordfile topics.txt --topicwordfile2 topics.vocab.txt

# four threads, note that reported time is cpu time, therefore divide by 2-3.5 to have better idea
# debugging mode off, after training accept doc from stdin for similarity testing
#export OMP_NUM_THREADS=4; cat data_uci/docword.nips.txt | python -u get_docword2oneline.py | ./lda_scvb0 -p -d data_uci/docword.nips.txt -v data_uci/vocab.nips.txt -s 10 -k 100 --doctopicfile doctopic.txt --topicwordfile topics.txt --topicwordfile2 topics.vocab.txt

export OMP_NUM_THREADS=4; ./lda_scvb0 -p -d data_uci/docword.nips.txt -v data_uci/vocab.nips.txt -s 10 -k 100 --doctopicfile doctopic.txt --topicwordfile topics.txt --topicwordfile2 topics.vocab.txt


