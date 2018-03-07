# Many birds, one stone
Finds out symptoms similar to a given symptom, from a symptom-disease data set.

This model is used to predict symptoms that are closely related to a given symptom. It can be used in cases (read apps) where the user enters a symptom, and a list of similar symptoms pop up, of which the user can select the ones he's suffering from, and these can be further fed into a model that can then predict the disease the person is suffering from, and redirect him to the associated specialist. The latter part isn't included here.

The data set contains a table of diseases and the associated symptoms. The model architecture is as follows:
 1) After preprocessing, make the data into the symptom-disease format from the existing disease-symptom format.
 2) Make symptoms the target words and the associated diseases the context words, and use this as the (target word, context word) pair for     skipgram generation.
 3) After assigning labels of 1 or 0 to the pairs, feed it into the Keras model, which generates new word vectors on top of existing GloVe     vectors
 4) Loop through the set of all symptoms in the data set to find out the cosine similarity between the embeddings of the given symptom and     current symptom in the loop, and then print out the symptoms with a high similrity score.
 
The '.npy' files are the new word vectors for the symptoms that have been trained for different epochs. The similarity_score value is a hyperparameter that needs to be tuned with the number of epochs.

The different csv files are :
 1) Dictionary, that shows all the symptoms and diseases and their corresponding indexes in the Keras model.
 2) Symptom Counts, which show the number of occurences of each symptom in the data set.
 3) Unrepresented Words, which shows the occurences of words that are not represented in the GloVe vectors.

The code is heavily documented, and all the details regarding the implementation of the model architecture can be found in it.
