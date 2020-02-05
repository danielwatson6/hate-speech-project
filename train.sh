source env/bin/activate

python run.py fit rnn mf_rnn twitter_mf
python run.py fit doc2vec_256_256 mf_doc2vec twitter_mf --hidden_sizes 256 256
python run.py fit doc2vec_512_512 mf_doc2vec twitter_mf --hidden_sizes 512 512
python run.py fit doc2vec_512_256 mf_doc2vec twitter_mf --hidden_sizes 512 256
python run.py fit doc2vec_256_512 mf_doc2vec twitter_mf --hidden_sizes 256 512

