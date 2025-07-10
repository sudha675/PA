import nltk
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

input="i am still learing python and i am enjoying it"

word=word_tokenize(input)

ps=PorterStemmer()

stemmer_word=[]
for token in word:
    stemmer_word.append(ps.stem(token))

print(stemmer_word)