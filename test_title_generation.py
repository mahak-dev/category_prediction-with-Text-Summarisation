
pip install tensorflow

import numpy as np
import pandas as pd
import pickle
from statistics import mode
import nltk
from nltk import word_tokenize
from nltk.stem import LancasterStemmer
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('punkt')
from nltk.corpus import stopwords
from tensorflow.keras.models import Model
from tensorflow.keras import models
from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer 
from tensorflow.keras.utils import plot_model
from tensorflow.keras.layers import Input,LSTM,Embedding,Dense,Concatenate,Attention
from sklearn.model_selection import train_test_split
from bs4 import BeautifulSoup



df=pd.read_csv("test.csv", nrows=10000)

df.drop_duplicates(subset=['Summary'],inplace=True)
df.dropna(axis=0,inplace=True)
input_data = df.loc[:,'Summary']
target_data = df.loc[:,'Headline']
target_data.replace('', np.nan, inplace=True)





input_texts=[]
target_texts=[]
input_words=[]
target_words=[]
contractions=pickle.load(open("contractions.pkl","rb"))['contractions']

stop_words=set(stopwords.words('english'))
stemm=LancasterStemmer()





def clean(texts,src):

  texts = BeautifulSoup(texts, "lxml").text 
  words=word_tokenize(texts.lower())
  words= list(filter(lambda w:(w.isalpha() and len(w)>=3),words))
  words= [contractions[w] if w in contractions else w for w in words ]
  if src=="inputs":
    words= [stemm.stem(w) for w in words if w not in stop_words]
  else:
    words= [w for w in words if w not in stop_words]
  return words



for in_txt,tr_txt in zip(input_data,target_data):
  in_words= clean(in_txt,"inputs")
  input_texts+= [' '.join(in_words)]
  input_words+= in_words
  tr_words= clean("sos "+tr_txt+" eos","target")
  target_texts+= [' '.join(tr_words)]
  target_words+= tr_words



input_words = sorted(list(set(input_words)))
target_words = sorted(list(set(target_words)))
num_in_words = len(input_words)
num_tr_words = len(target_words)

max_in_len = 30
max_tr_len = 10  
latent_dim = 300 
 
print("number of input words : ",num_in_words)
print("number of target words : ",num_tr_words)
print("maximum input length : ",max_in_len)
print("maximum target length : ",max_tr_len)



x_train,x_test,y_train,y_test=train_test_split(input_texts,target_texts,test_size=0.2,random_state=0) 



in_tokenizer = Tokenizer()
in_tokenizer.fit_on_texts(x_train)
tr_tokenizer = Tokenizer()
tr_tokenizer.fit_on_texts(y_train)
 

x_train= in_tokenizer.texts_to_sequences(x_train) 
y_train= tr_tokenizer.texts_to_sequences(y_train)


 
en_in_data= pad_sequences(x_train,  maxlen=max_in_len, padding='post') 
dec_data= pad_sequences(y_train,  maxlen=max_tr_len, padding='post')

dec_in_data = dec_data[:,:-1]

dec_tr_data = dec_data.reshape(len(dec_data),max_tr_len,1)[:,1:]


K.clear_session() 
latent_dim = 500
 
en_inputs = Input(shape=(max_in_len,)) 
en_embedding = Embedding(num_in_words+1, latent_dim)(en_inputs) 



en_lstm1= LSTM(latent_dim, return_state=True, return_sequences=True) 
en_outputs1, state_h1, state_c1= en_lstm1(en_embedding) 
 
en_lstm2= LSTM(latent_dim, return_state=True, return_sequences=True) 
en_outputs2, state_h2, state_c2= en_lstm2(en_outputs1) 
 
en_lstm3= LSTM(latent_dim,return_sequences=True,return_state=True)
en_outputs3 , state_h3 , state_c3= en_lstm3(en_outputs2)

en_states= [state_h3, state_c3]


 
dec_inputs = Input(shape=(None,)) 
dec_emb_layer = Embedding(num_tr_words+1, latent_dim) 
dec_embedding = dec_emb_layer(dec_inputs) 

dec_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
dec_outputs, *_ = dec_lstm(dec_embedding,initial_state=en_states) 



attention =Attention()
attn_out = attention([dec_outputs,en_outputs3])
 
merge=Concatenate(axis=-1, name='concat_layer1')([dec_outputs,attn_out])



dec_dense = Dense(num_tr_words+1, activation='softmax') 
dec_outputs = dec_dense(merge)



model = Model([en_inputs, dec_inputs], dec_outputs) 
model.summary()
plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)



model.compile( 
    optimizer="rmsprop", loss="sparse_categorical_crossentropy", metrics=["accuracy"] ) 
model.fit( 
    [en_in_data, dec_in_data],
    dec_tr_data, 
    batch_size=512, 
    epochs=2, 
    validation_split=0.1,
    )
 
model.save("s2s")



latent_dim=500
model = models.load_model("s2s")
 
en_outputs,state_h_enc,state_c_enc = model.layers[6].output
en_states=[state_h_enc,state_c_enc]
en_model = Model(model.input[0],[en_outputs]+en_states)



dec_state_input_h = Input(shape=(latent_dim,))
dec_state_input_c = Input(shape=(latent_dim,))
dec_hidden_state_input = Input(shape=(max_in_len,latent_dim))

dec_inputs = model.input[1]
dec_emb_layer = model.layers[5]
dec_lstm = model.layers[7]
dec_embedding= dec_emb_layer(dec_inputs)

dec_outputs2, state_h2, state_c2 = dec_lstm(dec_embedding, initial_state=[dec_state_input_h,dec_state_input_c])



attention = model.layers[8]
attn_out2 = attention([dec_outputs2,dec_hidden_state_input])
 
merge2 = Concatenate(axis=-1)([dec_outputs2, attn_out2])



dec_dense = model.layers[10]
dec_outputs2 = dec_dense(merge2)
 

dec_model = Model(
[dec_inputs] + [dec_hidden_state_input,dec_state_input_h,dec_state_input_c],
[dec_outputs2] + [state_h2, state_c2])



reverse_target_word_index = tr_tokenizer.index_word
reverse_source_word_index = in_tokenizer.index_word
target_word_index = tr_tokenizer.word_index
reverse_target_word_index[0]=' '

def decode_sequence(input_seq):
    en_out, en_h, en_c= en_model.predict(input_seq)

    target_seq = np.zeros((1, 1))
    target_seq[0, 0] = target_word_index['sos']

    stop_condition = False
    decoded_sentence = ""
    while not stop_condition: 
        output_words, dec_h, dec_c= dec_model.predict([target_seq] + [en_out,en_h, en_c])
        
        word_index = np.argmax(output_words[0, -1, :])
        text_word = reverse_target_word_index[word_index]
        decoded_sentence += text_word +" "

        if text_word == "eos" or len(decoded_sentence) > max_tr_len:
          stop_condition = True
        
        target_seq = np.zeros((1, 1))
        target_seq[0, 0] = word_index
        en_h, en_c = dec_h, dec_c
    
    return decoded_sentence



inp_review = input("Enter : ")
print("Review :",inp_review)
inp_review = clean(inp_review,"inputs")
inp_review = ' '.join(inp_review)
inp_x= in_tokenizer.texts_to_sequences([inp_review]) 
inp_x= pad_sequences(inp_x,  maxlen=max_in_len, padding='post')
 
summary=decode_sequence(inp_x.reshape(1,max_in_len))
if 'eos' in summary :
  summary=summary.replace('eos','')
print("\nPredicted summary:",summary);print("\n")

import pickle
from tensorflow.keras.models import load_model

with open('train_title_generation.pkl', 'wb') as tokenizer_file:
    pickle.dump(in_tokenizer, tokenizer_file)

model.save('seq2seq_model.h5')

model_info = {
    'max_in_len': max_in_len,
    'max_tr_len': max_tr_len,
    'latent_dim': latent_dim,
}

with open('model_info.pkl', 'wb') as model_info_file:
    pickle.dump(model_info, model_info_file)

