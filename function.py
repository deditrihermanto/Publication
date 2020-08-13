
# Defining a class
class InputSentence:
	def __init__(self, input_text, model_lstm, model_lstm_cnn, model_cnn_lstm):
		self.input_text = input_text
		self.model_lstm = model_lstm
		self.model_lstm_cnn = model_lstm_cnn
		self.model_cnn_lstm = model_cnn_lstm


	def get_input(self):
		print(self.input_text)
		
	def Predict(self):
			import re
			import nltk
			from nltk.tokenize import word_tokenize
			from tensorflow.keras.preprocessing.text import Tokenizer
			from tensorflow.keras.preprocessing.sequence import pad_sequences
			from tensorflow.keras.preprocessing import text, sequence

			##Casefolding
			data_test = self.input_text.lower()

			##Filtering
			data_test = re.sub(r'\d+', '', data_test)
			data_test = re.sub(r'[^\w]', ' ', data_test)

			##Tokenization
			data_test= word_tokenize(data_test)
			
			max_features = 20000
			max_text_length = 32

			#Create y variable from Text in Dataset
			sentence_data_test = data_test

			#Tokenize Data Test y_test
			data_test_tokenizer = text.Tokenizer(max_features)
			data_test_tokenizer.fit_on_texts(sentence_data_test)
			data_test_encode = data_test_tokenizer.texts_to_sequences([sentence_data_test])
			data_test_sequence = sequence.pad_sequences(data_test_encode, maxlen = max_text_length)
						
			import tensorflow as tf
			modelLSTM = tf.keras.models.load_model(self.model_lstm)
			modelLSTM_CNN = tf.keras.models.load_model(self.model_lstm_cnn)
			modelCNN_LSTM = tf.keras.models.load_model(self.model_cnn_lstm)
			
			predictLSTM = modelLSTM.predict(data_test_sequence, verbose=0, batch_size=128)
			predictLSTM_CNN = modelLSTM_CNN.predict(data_test_sequence, verbose=0, batch_size=128)
			predictCNN_LSTM = modelCNN_LSTM.predict(data_test_sequence, verbose=0, batch_size=128)
			
			data_testhasil1 = ['negative' if x < .51 else 'positive' for x in predictLSTM]
			data_testhasil2 = ['negative' if x < .51 else 'positive' for x in predictLSTM_CNN]
			data_testhasil3 = ['negative' if x < .51 else 'positive' for x in predictCNN_LSTM]

			print(f'Berita "{self.input_text}"')
			print(f'Hasil Analisis Sentimen Model LSTM pada berita diatas adalah "{data_testhasil1[0]}"')
			print(f'Hasil Analisis Sentimen Model LSTM-CNN pada berita diatas adalah "{data_testhasil2[0]}"')
			print(f'Hasil Analisis Sentimen Model CNN-LSTM pada berita diatas adalah "{data_testhasil3[0]}"')
			