import streamlit as st
from keras.models import load_model
import re
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from sklearn.preprocessing import LabelEncoder
import json
import pandas as pd
import numpy as np
import random

model = load_model("./models/smcb_v4.h5", compile=False)
word_index = {'i': 1, 'you': 2, 'what': 3, 'mental': 4, 'to': 5, 'about': 6, 'me': 7, 'health': 8, 'am': 9, 'can': 10, 'feel': 11, 'are': 12, 'do': 13, 'is': 14, 'my': 15, 'a': 16, "don't": 17, 'for': 18, 'the': 19, 'so': 20, 'have': 21, "i'm": 22, 'want': 23, 'how': 24, 'of': 25, 'that': 26, 'know': 27, "can't": 28, 'it': 29, 'depression': 30, 'myself': 31, 'help': 32, 'who': 33, 'tell': 34, 'more': 35, 'need': 36, 'like': 37, 'where': 38, 'learn': 39, 'if': 40, 'there': 41, 'good': 42, 'no': 43, 'not': 44, 'talk': 45, "you're": 46, 'would': 47, 'problems': 48, 'and': 49, 'illness': 50, 'ok': 51, 'much': 52, 'name': 53, 'go': 54, 'lonely': 55, 'stressed': 56, 
              'because': 57, 'away': 58, 'some': 59, 'all': 60, 'else': 61, 'exams': 62, 'something': 63, 'therapy': 64, 'find': 65, 'difference': 66, 'between': 67, 'bye': 68, 'thank': 69, "that's": 70, 'very': 71, 'nothing': 72, 'yourself': 73, 'your': 74, 'should': 75, "what's": 76, 'support': 77, 'out': 78, 'depressed': 79, 'think': 80, 'yeah': 81, 'just': 82, 'sleep': 83, 'sounds': 84, 'this': 85, 'died': 86, 'passed': 87, 'in': 88, 'going': 89, 'hate': 90, 'right': 91, 'advice': 92, 'does': 93, 'treatment': 94, 'hi': 95, 'hey': 96, 'anyone': 97, 'see': 98, 'well': 99, 'thanks': 100, 'created': 101, 'were': 102, 'could': 103, 'please': 104, 'sad': 105, 'anything': 106, 
              'useless': 107, 'sense': 108, 'anymore': 109, 'happy': 110, 'fine': 111, 'yes': 112, 'really': 113, 'anxious': 114, 'up': 115, 'suffering': 116, 'from': 117, "haven't": 118, 'days': 119, 'scared': 120, 'someone': 121, 'family': 122, 'say': 123, 'be': 124, 'kill': 125, "i've": 126, 'friends': 127, 'joke': 128, 'another': 129, 'already': 130, 'why': 131, 'wrong': 132, 'dumb': 133, 'any': 134, 'probably': 135, 'guess': 136, 'useful': 137, 'better': 138, 'on': 139, 'fact': 140, 'define': 141, 'therapist': 142, 'causes': 143, 'professional': 144, 'or': 145, 'child': 146, 'professionals': 147, 'types': 148, 'sadness': 149, 'hello': 150, 'howdy': 151, 'hola': 152, 
              'bonjour': 153, 'konnichiwa': 154, 'guten': 155, 'tag': 156, 'ola': 157, 'morning': 158, 'afternoon': 159, 'evening': 160, 'night': 161, 'later': 162, 'goodbye': 163, 'au': 164, 'revoir': 165, 'sayonara': 166, 'then': 167, 'fare': 168, 'thee': 169, 'helpful': 170, 'than': 171, 'call': 172, 'made': 173, 'by': 174, 'give': 175, 'hand': 176, 'feeling': 177, 'down': 178, 'empty': 179, 'stuck': 180, 'still': 181, 'burned': 182, 'worthless': 183, 'one': 184, 'likes': 185, 'makes': 186, 'take': 187, 'great': 188, 'today': 189, 'cheerful': 190, 'oh': 191, 'okay': 192, 'nice': 193, 'whatever': 194, 'k': 195, 'stay': 196, 'bring': 197, 'open': 198, 'shut': 199, 'insominia': 200, 
              'insomnia': 201, 'slept': 202, 'last': 203, 'seem': 204, 'had': 205, 'proper': 206, 'past': 207, 'few': 208, 'awful': 209, 'way': 210, 'mom': 211, 'brother': 212, 'dad': 213, 'sister': 214, 'friend': 215, 'understand': 216, 'robot': 217, 'possibly': 218, 'through': 219, 'nobody': 220, 'understands': 221, 'thought': 222, 'killing': 223, 'die': 224, 'commit': 225, 'suicide': 226, 'trust': 227, 'relationship': 228, 'boyfriend': 229, 'girlfriend': 230, 'money': 231, 'financial': 232, 'told': 233, 'mentioned': 234, 'repeating': 235, 'saying': 236, "doesn't": 237, 'make': 238, 'response': 239, 'answer': 240, 'stupid': 241, 'crazy': 242, 'live': 243, 'location': 244, "let's": 245, 
              'we': 246, 'ask': 247, 'approaching': 248, 'prepared': 249, 'enough': 250, 'sure': 251, 'deserve': 252, 'break': 253, 'absolutely': 254, 'hmmm': 255, 'did': 256, 'said': 257, 'alot': 258, 'now': 259, 'again': 260, "i'll": 261, 'continue': 262, 'practicing': 263, 'meditation': 264, 'focus': 265, 'control': 266, 'interested': 267, 'learning': 268, 'important': 269, 'importance': 270, 'mentally': 271, 'ill': 272, 'mean': 273, 'affect': 274, 'warning': 275, 'signs': 276, 'people': 277, 'with': 278, 'recover': 279, 'appears': 280, 'symptoms': 281, 'disorder': 282, 'options': 283, 'available': 284, 'become': 285, 'involved': 286, 'get': 287, 'before': 288, 'starting': 289, 'new': 290, 
              'medication': 291, 'different': 292, 'group': 293, 'prevent': 294, 'cures': 295, 'cure': 296, 'worried': 297, 'unwell': 298, 'maintain': 299, 'social': 300, 'connections': 301, 'anxiety': 302, 'stress': 303}

with open("./intents.json", "r") as file:
    data = json.load(file)

df = pd.DataFrame(data['intents'])
dic = {"tag":[], "patterns":[], "responses":[]}
for i in range(len(df)):
    ptrns = df[df.index == i]['patterns'].values[0]
    rspns = df[df.index == i]['responses'].values[0]
    tag = df[df.index == i]['tag'].values[0]
    for j in range(len(ptrns)):
        dic['tag'].append(tag)
        dic['patterns'].append(ptrns[j])
        dic['responses'].append((rspns))
        
df = pd.DataFrame.from_dict(dic)

lbl_enc = LabelEncoder()
lbl_enc.fit(df['tag'])

def custom_txt_to_seq(text:str):
    sequence = [[]]
    for i in text[0].split(" "):
        if len(text[0].split(" ")) <= 1:
            sequence[0].append(0)
            if i not in list(word_index.keys()):
                sequence[0].append(0)
            else:
                sequence[0].append(word_index[i])
        elif i not in list(word_index.keys()):
                sequence[0].append(0)
        else:
            sequence[0].append(word_index[i])
    return sequence

def generate_answer(pattern): 
    text = []
    txt = re.sub('[^a-zA-Z\']', ' ', pattern)
    txt = txt.lower()
    txt = txt.split()
    txt = " ".join(txt)
    text.append(txt)
    print(text)
    # x_test = tokenizer.texts_to_sequences(text)
    x_test = custom_txt_to_seq(text)
    print(x_test)
    x_test = np.array(x_test).squeeze()
    x_test = pad_sequences([x_test], padding='post', maxlen=18)
    y_pred = model.predict(x_test)
    y_pred = y_pred.argmax()
    tag = lbl_enc.inverse_transform([y_pred])[0]
    responses = df[df['tag'] == tag]['responses'].values[0]
    return random.choice(responses)

st.title("Medical Chat Bot")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("How can I help you?"):
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    response = f"{generate_answer(prompt)}"
    with st.chat_message("assistant"):
        st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})
