import streamlit as st
import pandas as pd
import torch
from transformers import RobertaForSequenceClassification, AutoTokenizer
import numpy as np
import requests
from bs4 import BeautifulSoup
# import nltk
from underthesea import word_tokenize

# nltk.download('punkt')

model = RobertaForSequenceClassification.from_pretrained("wonrax/phobert-base-vietnamese-sentiment")

tokenizer = AutoTokenizer.from_pretrained("wonrax/phobert-base-vietnamese-sentiment", use_fast=False)

# Ta cần custom stopword tiếng Việt do thư viện này không có sẵn stopword list cho tiếng Việt
# with open('vietnamese-stopwords.txt', encoding='utf-8') as fr:
#     stopwords = fr.readlines()

sentiment_labels = ["Tiêu cực", "Tích cực", "Trung lập"]
sentiment_confidence = 0.95

baodautu_url = 'https://baodautu.vn/'
cafef_url = 'https://cafef.vn/'
vneconomy_url = 'https://vneconomy.vn/'
vnexpress_url = 'https://vnexpress.net/'

with open('topics_keywords.txt', encoding='utf-8') as fr:
    lines = fr.readlines()
    lines = [line.strip() for line in lines]    
    topics_keywords = []
    for line in lines:
        topic = line.split(':')[0].lower()
        keywords = line.split(':')[1].split(',')
        keywords = [keyword.lower() for keyword in keywords]
        topics_keywords.append((topic, keywords))

with open('banks.txt', encoding='utf-8') as fr:
    lines = fr.readlines()
    lines = [line.strip() for line in lines]    
    banks_keywords = []
    for line in lines:
        topic = line.split(':')[0]
        keywords = line.split(':')[1].split(',')
        banks_keywords.append((topic, keywords))

def text_tokenize_punkt(text, lower=True):
    # Tải các nguồn dữ liệu cần thiết cho việc tokenize
    if lower:
        text = text.lower()
    
    # Sử dụng nltk.tokenize để thực hiện tokenize
    tokens = word_tokenize(text)

    sentence = ' '.join(tokens)
    
    return sentence

def normalize_sentiment(sentiment_array):
    # Mảng ban đầu
    arr = np.array(sentiment_array)

    # Chuyển các giá trị > 0.5 thành 1, các giá trị khác thành 0
    result = np.where(arr > 0.5, 1, 0)

    return result

def get_sentiment_from_phobert(text):
    if text != '':
        input_ids = torch.tensor([tokenizer.encode(text)])
        if input_ids.shape[1] <= 258:

            with torch.no_grad():
                out = model(input_ids)
                return out.logits.softmax(dim=-1).tolist()
                # Output:
                # [[0.002, 0.988, 0.01]]
                #     ^      ^      ^
                #    NEG    POS    NEU
        else:
            return np.zeros((1, 3))
    else:
        return np.zeros((1, 3))
    
def get_article_sentiment_and_topics_and_banks(title, abstract, content):
    total_sentiment = np.zeros((1, 3))
    all_banks, all_topics = [], []

    # Match keywords and topics in the title
    matched_topics_and_keywords_title = get_topics_and_keywords(title)
    all_topics = all_topics + [m[0] for m in matched_topics_and_keywords_title]
    # Match banks in the title
    matched_banks_title = get_banks(title)
    all_banks = all_banks + [m[0] for m in matched_banks_title]
    # Get sentiment from the title
    title_sentiment_result = get_sentiment_from_phobert(title)
    # Update the total sentiment
    total_sentiment = total_sentiment + normalize_sentiment(title_sentiment_result)

    st.write('Title: ' + title)
    st.write('Chủ đề:' + str(matched_topics_and_keywords_title))  
    st.write('Ngân hàng:' + str(matched_banks_title))                
    st.markdown(generate_sentiment_markdown(title_sentiment_result), unsafe_allow_html=True)
    st.write('--------------------------------')

    # Match keywords and topics in the abstract
    matched_topics_and_keywords_abstract = get_topics_and_keywords(abstract)
    all_topics = all_topics + [m[0] for m in matched_topics_and_keywords_abstract]
    # Match banks in the abstract
    matched_banks_abstract = get_banks(abstract)
    all_banks = all_banks + [m[0] for m in matched_banks_abstract]
    # Get sentiment from the abstract
    abstract_sentiment_result = get_sentiment_from_phobert(abstract)
    # Update the total sentiment
    total_sentiment = total_sentiment + normalize_sentiment(abstract_sentiment_result)

    st.write('Abstract: ' + abstract)
    st.write('Chủ đề:' + str(matched_topics_and_keywords_abstract))  
    st.write('Ngân hàng:' + str(matched_banks_abstract))  
    st.markdown(generate_sentiment_markdown(abstract_sentiment_result), unsafe_allow_html=True)
    st.write('--------------------------------')

    paragraphs = content.split('\n')
    paragraphs = [paragraph for paragraph in paragraphs if paragraph != '']
    for paragraph in paragraphs:
        matched_topics_and_keywords_paragraph = get_topics_and_keywords(paragraph)
        all_topics = all_topics + [m[0] for m in matched_topics_and_keywords_paragraph]
        matched_banks_paragraph = get_banks(paragraph)
        all_banks = all_banks + [m[0] for m in matched_banks_paragraph]
        paragraph_sentiment_array = get_sentiment_from_phobert(paragraph)
        total_sentiment = total_sentiment + normalize_sentiment(paragraph_sentiment_array)

        # st.write(paragraph)
        # st.write('Chủ đề:' + str(matched_topics_and_keywords_paragraph)) 
        # st.write('Ngân hàng:' + str(matched_banks_paragraph))  
        # st.markdown(generate_sentiment_markdown(paragraph_sentiment_array), unsafe_allow_html=True)
        # st.write('--------------------------------')

    if (total_sentiment - np.zeros((1, 3))).any():
        max_index = np.argmax(total_sentiment[0])
        max_label = sentiment_labels[max_index]
        if max_label == 'Tích cực':
            max_label_markdown = '**:blue[' + max_label + ']**'
        elif max_label == 'Tiêu cực':
            max_label_markdown = '**:red[' + max_label + ']**'
        else:
            max_label_markdown = max_label

        st.markdown('_Sắc thái chung: ' + max_label_markdown + '_')
        st.write('Số đoạn có sắc thái tiêu cực: ' + str(total_sentiment[0][0]))
        st.write('Số đoạn có sắc thái tích cực: ' + str(total_sentiment[0][1]))
        st.write('Số đoạn có sắc thái trung lập: ' + str(total_sentiment[0][2]))
    else:
        st.markdown('_Sắc thái: Không xác định do đoạn văn quá dài')
    st.write('----------------------------------------------------------------')
    st.write('Các chủ đề được nhắc đến trong bài: ')
    st.write(set(all_topics))
    st.write('----------------------------------------------------------------')
    st.write('Các ngân hàng được nhắc đến trong bài: ')
    st.write(set(all_banks))

def get_banks(text):
    text_preprocessed = text_tokenize_punkt(text, lower=False)
    matched_banks = []
    for bank_keywords in banks_keywords:
        matched_keywords = []
        candidate_keywords = bank_keywords[1]
        candidate_bank = bank_keywords[0]
        for keyword in candidate_keywords:
            if keyword in text_preprocessed:
                matched_keywords.append(keyword)
        if len(matched_keywords) > 0:
            matched_banks.append((candidate_bank, matched_keywords))
    return matched_banks

def get_topics_and_keywords(text):
    text_preprocessed = text_tokenize_punkt(text)
    matched_topics_and_keywords = []
    for topic_keywords in topics_keywords:
        matched_keywords = []
        candidate_keywords = topic_keywords[1]
        candidate_topic = topic_keywords[0]
        for keyword in candidate_keywords:
            if keyword in text_preprocessed:
                matched_keywords.append(keyword)
        if len(matched_keywords) > 0:
            matched_topics_and_keywords.append((candidate_topic, matched_keywords))
    return matched_topics_and_keywords
    
def get_vnexpress_content_from_url(url):
    news = requests.get(url)
    soup = BeautifulSoup(news.content, "html.parser")
    try:
      title = soup.find("h1", class_="title-detail").text.strip()
    except:
      title = ''
    try:
      abstract = soup.find("p", class_="description").text.strip()
    except:
      abstract = ''

    body = soup.find("article", class_="fck_detail")
    try:
      content = ''
      i = 0
      while True:
        try: 
          content = content + body.findChildren("p", recursive=True)[i].text + '\n'
          i = i + 1
        except:
            break
    except:
      content = ''
    
    return title, abstract, content

def generate_sentiment_markdown(sentiment_array):
    if (sentiment_array - np.zeros((1, 3))).any():
        max_value = max(sentiment_array[0])
        max_index = sentiment_array[0].index(max_value)
        max_label = sentiment_labels[max_index]
        if max_label == 'Tích cực':
            max_label_markdown = '**:blue[' + max_label + ']**'
        elif max_label == 'Tiêu cực':
            max_label_markdown = '**:red[' + max_label + ']**'
        else:
            max_label_markdown = max_label

        if max_value < sentiment_confidence:
            max_value_markdown = '**:green[' + str(max_value) + ']**'
        else: 
            max_value_markdown = str(max_value)

        return '_Sắc thái: ' + max_label_markdown + ' - Độ tin cậy: ' + max_value_markdown + '_'
    else:
        return '_Sắc thái: Không xác định do đoạn văn quá dài'


st.set_page_config(layout="wide")

# Set header of the page
st.header('Phân tích quan điểm báo chí tài chính ngân hàng')

# Split tabs
tab1, tab2, tab3 = st.tabs(["Sentiment from text", "Sentiment from URL", "Summary"])

# Sentiment from text
with tab1:
    with st.form("sentiment_from_text_form"):        
        input_text = st.text_area('Text to analyze', height = 80)
        submitted = st.form_submit_button("Submit")
        if submitted:
            paragraphs = input_text.split('\n')
            paragraphs = [paragraph for paragraph in paragraphs if paragraph != '']
            for paragraph in paragraphs:
                matched_topics_and_keywords_paragraph = get_topics_and_keywords(paragraph)
                matched_banks_paragraph = get_banks(paragraph)     
                result_array = get_sentiment_from_phobert(paragraph)
                st.write(paragraph)
                st.write('Chủ đề:' + str(matched_topics_and_keywords_paragraph))  
                st.write('Ngân hàng:' + str(matched_banks_paragraph))  
                st.markdown(generate_sentiment_markdown(result_array), unsafe_allow_html=True)

with tab2:
    with st.form("sentiment_from_URL_form"): 
        input_url = st.text_input('VnExpress URL')
        submitted = st.form_submit_button("Submit")
        if submitted:
            if input_url.startswith(vnexpress_url):
                # get title, abstract, content by crawling through the page
                title, abstract, content = get_vnexpress_content_from_url(input_url)
                with st.spinner('Wait for it...'):
                    get_article_sentiment_and_topics_and_banks(title, abstract, content)
                
            else:
                st.write('VnExpress cơ mà, nhập lại đê')
            
with tab3:
    st.header("Tổng hợp tin tức báo chí")
    news_df = pd.read_csv('data/news_dataset.csv')
    st.dataframe(news_df)   