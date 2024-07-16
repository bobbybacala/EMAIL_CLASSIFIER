

# importing necessary libraries
import streamlit as st
import pickle
import nltk
import string
from streamlit_option_menu import option_menu
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Ensure NLTK resources are downloaded
nltk.download('punkt')
nltk.download('stopwords')

# obj of PorterStemmer
ps = PorterStemmer()

# defining the function to preprocess the text  
def text_transform(text):
    # to lowecase
    text = text.lower()
    
    # cut the sentences in words
    text = nltk.word_tokenize(text)
    
    # make a list and append only the apla-numeric characters in it(REMOVES THE SPECIAL CHARS)
    y = []
    for i in text:
        if i.isalnum():
            y.append(i)
            
    # clone the list back in text
    text = y[:]
    y.clear()
    
    # remove stop words and punctuation
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    # cloning
    text = y[:]
    y.clear()
    
    # stemming the words
    for i in text:
        stemmed_word = ps.stem(i)
        y.append(stemmed_word)
        
    # return 'y' list in the form of a string
    return " ".join(y)


# -----------------------------------------------------------------------------------------------------------------------

# load vectorizer and pickle model for class
tfidf_class = pickle.load(open('vectoriser_class.pkl', 'rb'))
model_class = pickle.load(open('model_class.pkl', 'rb'))

# load vectoriser and model for ham and spam
tfidf_ham_spam = pickle.load(open('vectoriser_ham_spam.pkl','rb'))
model_ham_spam = pickle.load(open('model_ham_spam.pkl','rb'))

# load vectoriser and model for priority 
tfidf_priority = pickle.load(open('vectoriser_priority.pkl', 'rb'))
model_priority = pickle.load(open('model_priority.pkl', 'rb'))

#-------------------------------------------------------------------------------------------------------------------------

# making the layout wide
st. set_page_config(layout="wide")

# creating a navigation menu
selected = option_menu(
    menu_title=None,
    options = ['Home', 'About us', 'Contact' ],
    orientation='horizontal',
)

if selected == 'Home':
    
    st.title('Type of Email Classifier')

    mail_body_input = st.text_input('Enter your mail message')

    # button to predict
    if st.button('Predict'):
        # 1. Preprocess the Text
        transformed_mail = text_transform(mail_body_input)

        # 2. Vectorize the Text
        vector_mail = tfidf_class.transform([transformed_mail])
        vector_mail_ham_spam = tfidf_ham_spam.transform([transformed_mail])
        vector_priority = tfidf_priority.transform([transformed_mail])

        # 3. Convert the sparse matrix to a dense array
        vector_mail_dense = vector_mail.toarray()
        vector_mail_ham_spam_dense = vector_mail_ham_spam.toarray()
        vector_mail_priority_dense = vector_priority.toarray()

        # 4. Predict
        result = model_class.predict(vector_mail_dense)[0]
        result_2 = model_ham_spam.predict(vector_mail_ham_spam_dense)[0]
        result_3 = model_priority.predict(vector_mail_priority_dense)[0]

        # 5. Display
        st.header('Type of the mail:')
        if result == 0:
            st.markdown('<p style="font-size:24px;">Advertisement</p>', unsafe_allow_html=True)
        elif result == 1:
            st.markdown('<p style="font-size:24px;">Customer Support</p>', unsafe_allow_html=True)
        elif result == 2:
            st.markdown('<p style="font-size:24px;">Finance</p>', unsafe_allow_html=True)
        elif result == 3:
            st.markdown('<p style="font-size:24px;">HR</p>', unsafe_allow_html=True)
        elif result == 4:
            st.markdown('<p style="font-size:24px;">Operations</p>', unsafe_allow_html=True)
            
        # 6. Display if Spam
        st.header('Target:')
        if result_2 == 0:
            st.markdown('<p style="font-size:24px;">Ham</p>', unsafe_allow_html=True)
        if result_2 == 1:
            st.markdown('<p style="font-size:24px;">Mail likely to be Spam</p>', unsafe_allow_html=True)
            
        # 7. Display Priority
        st.header('Priority: ')
        if result_3 == 0:
            st.markdown('<p style="font-size:24px;">High, Seems like an Important Mail</p>', unsafe_allow_html=True)
        elif result_3 == 1:
            st.markdown('<p style="font-size:24px;">Low, can be ignored</p>', unsafe_allow_html=True)
        elif result_3 == 2:
            st.markdown('<p style="font-size:24px;">Medium, Go throught the mail, could be important</p>', unsafe_allow_html=True)


if selected == 'About us':
    st.title('Our Team')
    
    st.markdown('''<style>
                    @import url('https://fonts.googleapis.com/css2?family=Montserrat&display=swap');
                    p { 
                        font-family: 'Montserrat', sans-serif; 
                        font-size: 24px; 
                    }
                  </style>''', unsafe_allow_html=True)
    st.markdown('<p>Susmit Bahadkar SY IT, VIT, Pune</p>', unsafe_allow_html=True)
    st.markdown('<p>Parth Bhalerao SY IT, VIT, Pune</p>', unsafe_allow_html=True)
    st.markdown('<p>Abhilash Baviskar SY IT, VIT, Pune</p>', unsafe_allow_html=True)
    st.markdown('<p>Jaywant Avhad SY IT, VIT, Pune</p>', unsafe_allow_html=True)
    st.markdown('<p>Aakash Chimkar SY IT, VIT, Pune</p>', unsafe_allow_html=True)
    st.write('')
    st.write('<p>A team of 5 passionate Engineering Students, having curious minds, eagerness for problem Solving.</p>', unsafe_allow_html=True)
    
if selected == 'Contact':
    st.title('Contact Us')
    
# a Custom footer using CSS
custom_ft = """
    <style>
        .footer{
            left: 0
            bottom: 0
            position: fixed;
            width: 100%;
            color: gray;
            text-align: center;
            font-size: 12px;
        }
    </style>
    <div class="footer">
        <p>© 2024 Copyright Batch 1 Group 2</p>
    </div>
        """
st.markdown(custom_ft, unsafe_allow_html=True)