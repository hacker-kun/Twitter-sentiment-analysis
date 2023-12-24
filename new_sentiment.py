import streamlit as st
from streamlit_option_menu import option_menu
from streamlit_lottie import st_lottie
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import requests



import re
import textblob
from textblob import TextBlob
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

import json

from io import StringIO

st.set_option('deprecation.showPyplotGlobalUse', False)

STYLE = st.markdown("""
<style>
div.stButton > button:first-child{
    background-color: #56a0d3;
    color:#ffffff;
    border: 2px solid red;
    height: 50px;
    width: 50%;

}
div.stButton >button:hover {
    background-color:#FF0000;
    color:##ff99ff;
}
}
</style> """, unsafe_allow_html=True)

# function for twitter animation


def load_lottiefile(filepath: str):
    with open(filepath, "r") as f:
        return json.load(f)

# function for snow flakes


def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


local_css(r"C:\Users\Rohit\Downloads\SHWETANK DOWNLOAD\Twitter-Sentiment-Analysis-main\Twitter-Sentiment-Analysis\style.css")

# load animation
animation_symbol = "❄"

st.markdown(f""" 
<div class="snowflake"> {animation_symbol}</div>
<div class="snowflake"> {animation_symbol}</div>
<div class="snowflake"> {animation_symbol}</div>
<div class="snowflake"> {animation_symbol}</div>
<div class="snowflake"> {animation_symbol}</div>
<div class="snowflake"> {animation_symbol}</div>
<div class="snowflake"> {animation_symbol}</div>
<div class="snowflake"> {animation_symbol}</div>
""", unsafe_allow_html=True)


def get_tweets_from_github(url, df=None, count=None):
    
    if df is None:
        df = pd.DataFrame(columns=['Tweet_ID', 'Username', 'Text', 'Retweets', 'Likes', 'Timestamp'])
        try:
            response = requests.get(url)
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            print(f"Error: {e}")
            return df  # Return the original DataFrame if there's an error

    # Read CSV data from the GitHub raw content URL
        csv_data = response.text.strip()
        if not csv_data:
            print("The CSV file is empty.")
            return df

    # Parse CSV data into a DataFrame
        data = pd.read_csv(StringIO(csv_data))

    # Extract and format required columns
        df['Tweet_ID'] = data['Tweet_ID']
        df['Username'] = data['Username']
        df['Text'] = data['Text']
        df['Retweets'] = data['Retweets']
        df['Likes'] = data['Likes']
    # df['Timestamp'] = pd.to_datetime(data['Timestamp'], format='%d-%m-%Y %H:%M') # Assuming 'Timestamp' is in datetime format

        df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')
        format1 = '%d-%m-%Y %H:%M'
        format2 = '%Y-%m-%d %H:%M:%S%z'

        df['Timestamp'] = df['Timestamp'].combine_first(
        pd.to_datetime(df['Timestamp'], format=format2, errors='coerce')
    )

    # Save DataFrame to CSV (optional)
        df.to_csv("TweetDataset1.csv", index=False)

    # Return the updated DataFrame
        return df

github_raw_url = "https://raw.githubusercontent.com/hacker-kun/Dataset/main/twdataset1.csv"
# main function


def main():
    html_temp = """
    <div style="background-color:Red;"><p style="color:white;font-size:40px;padding:10px"> Live Twitter Sentiment Analysis 😊🙂 </p></div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)

    df = pd.DataFrame(columns=['Tweet_ID', 'Username',
                      'Text', 'Retweets', 'Likes', 'Timestamp'])

    github_raw_url = "https://raw.githubusercontent.com/hacker-kun/Dataset/main/twdataset1.csv"

    # Call the function with the GitHub raw URL, existing DataFrame 'df', and desired count
    df = get_tweets_from_github(github_raw_url, df, count=None)

    

# function to clean the tweets


def clean_tweets(text):
    # Using regular expressions to substitute or remove specific patterns in the tweet text
    cleaned_tweet = re.sub(
        '(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|([RT])',
        ' ',
        str(text).lower()
    )

    # Split the cleaned tweet into words and join them back into a string
    cleaned_tweet = ' '.join(cleaned_tweet.split())

    return cleaned_tweet

# function to analyze sentiments


def sentiment_analyze(tweet):
        analysis = TextBlob(tweet)
        if analysis.sentiment.polarity > 0:
            return 'positive😊'
        elif analysis.sentiment.polarity == 0:
            return 'neutral🙂'
        else:
            return 'negative😑'

# Function to preprocess data for wordcloud


def prepcloud(Topic_text, Topic):
    Topic = str(Topic).lower()
    Topic = ' '.join(re.sub('[^0-9A-Za-z \t]', ' ', Topic).split())
    Topic = re.split("\s+", str(Topic))
    stopwords = set(STOPWORDS)
    # Add our topic in Stopwords, so it doesnt appear in wordCloud
    stopwords.update('Topic')

    text_new = " ".join(txt for txt in Topic_text.split()
                        if txt not in stopwords)
    return text_new


# function to extract tweets from twitter handle
df1 = pd.DataFrame(columns=['Username', 'Text',
                   'Retweets', 'Likes', 'Timestamp'])


def get_tweets_from_user(df, username,count_tweet=200):

    if df is None:
        return pd.DataFrame()

    for i in range(min(count_tweet, len(df))):
        tweet = df.iloc[i]

        if tweet['Username'] == username:
         df1.loc[i, 'Username'] = tweet['Username']
         df1.loc[i, 'Text'] = tweet['Text']
         df1.loc[i, 'Retweets'] = tweet['Retweets']
         df1.loc[i, 'Likes'] = tweet['Likes']
         df1.loc[i, 'Timestamp'] = tweet['Timestamp']

        # Optional: Save DataFrame to CSV after each iteration if needed
        df1.to_csv("TweetDataset2.csv", index=False)

    return df1


# animation picture
coding = load_lottiefile(r'C:\Users\Rohit\Downloads\SHWETANK DOWNLOAD\Twitter-Sentiment-Analysis-main\Twitter-Sentiment-Analysis\twitter-icon1.json')
st_lottie(coding, height=400)

# sentence -level analysis
st.subheader("Sentence-Level Analysis:")
text = str(st.text_input("Enter a Sentence"))
blob = TextBlob(text)
if blob.sentiment.polarity > 0:
        text_sentiment = "Positive😊"
elif blob.sentiment.polarity == 0:
        text_sentiment = "Neutral🙂"
else:
        text_sentiment = "Negative😑"
if len(text) > 0:
        st.write("Sentiment is : {}".format(text_sentiment))

# collect input from user:
st.subheader("Select a 'Topic' or '# hashtag' which you'd like to get the sentiment analysis on:")
Topic = str(st.text_input("Enter the Topic you are interested in (Press Enter once done)"))

df=None

if len(Topic) > 0:
    # call the function to extract the data
    with st.spinner("Please wait Tweets are being extracted"):
        df = get_tweets_from_github(github_raw_url, df, count=None)
        st.success("Tweets have been Extracted !!!")

    # call the function to get clean tweets
    df['Clean Tweets'] = df['Text'].apply(lambda x: clean_tweets(x))

    # call the function to analyze tweets
    df['Sentiment'] = df['Text'].apply(lambda x: sentiment_analyze(x))

    # Filter the DataFrame based on the specified topic
    topic_filtered_df = df[df['Text'].str.contains(Topic, case=False)]

    # Overall Summary
    st.write("Total tweets extracted for topic {}: {}".format(Topic, len(topic_filtered_df)))
    st.write("Total Positive Tweets: {}".format(len(topic_filtered_df[topic_filtered_df['Sentiment'] == 'positive😊'])))
    st.write("Total Neutral Tweets: {}".format(len(topic_filtered_df[topic_filtered_df['Sentiment'] == 'neutral🙂'])))
    st.write("Total Negative Tweets: {}".format(len(topic_filtered_df[topic_filtered_df['Sentiment'] == 'negative😑'])))

    # see the Extracted data
    if st.button("See the Extracted Data for {}:".format(Topic)):
        st.success("Below is the Extracted Data")
        st.write(topic_filtered_df.head(50))

        # get the count plot
        if st.button('Get Count Plot for Different Sentiments'):
            st.success("Generating a Count Plot")
            st.subheader("Count Plot for Different Sentiments")
            st.write(sns.countplot(x=df['Sentiment']))
            st.pyplot()

        # pie chart
        if st.button("Get Pie Chart for Different Sentiments"):
            st.success("Generating a Pie Chart")
            a=len(df[df['Sentiment']=='positive😊'])
            b=len(df[df['Sentiment']=='negative😑'])
            c=len(df[df['Sentiment']=='neutral🙂'])
            d=np.array([a,b,c])
            explode=(0.1,0.1,0.1)
            st.write(plt.pie(d,labels=['Positive','Negative','Neutral'],shadow=True,autopct='%1.2f%%',explode=explode))
            st.pyplot()
            
       # Wordcloud for Positive Tweets only
        if st.button("Get Word Cloud for all Positive Tweets about {}".format(Topic)):
           st.success("Generating a WordCloud for all Positive Tweets about {}".format(Topic))
           positive_text = " ".join(review for review in df[df['Sentiment']=='positive😊']['Clean Tweets'])
           stopwords = set(STOPWORDS)
           text_new_positive = prepcloud(positive_text, Topic)
           wordcloud_positive = WordCloud(stopwords=stopwords, max_words=800, max_font_size=70).generate(text_new_positive)
           st.write(plt.imshow(wordcloud_positive, interpolation='bilinear'))
           plt.axis("off")
           st.pyplot()

        

        # Wordcloud for Negative Tweets only
        if st.button("Get Word Cloud for all Negative Tweets about {}".format(Topic)):
           st.success("Generating a WordCloud for all Negative Tweets about {}".format(Topic))
           negative_text = " ".join(review for review in df[df['Sentiment']=='negative😑']['Clean Tweets'])
           stopwords = set(STOPWORDS)
           text_new_negative = prepcloud(negative_text, Topic)
           wordcloud_negative = WordCloud(stopwords=stopwords, max_words=800, max_font_size=70).generate(text_new_negative)
           st.write(plt.imshow(wordcloud_negative, interpolation='bilinear'))
           plt.axis("off")
           st.pyplot() 
            
st.subheader("Select a 'Twitter Handle' on whom tweets you'd like to get the sentiment analysis on:")
user_handle=str(st.text_input("Enter the Twitter handle (Press Enter once done)"))
if len(user_handle)>0:
        # call the function to extract the data
        with st.spinner("Please wait Tweets are being extracted"):
            # Assuming df is your main DataFrame containing tweets
            df_user_tweets = get_tweets_from_user(df, user_handle)
        
            st.success("Tweets have been Extracted !!!")

        #call the function to get clean tweets
        df1['Clean Tweets']=df1['Text'].apply(lambda x:clean_tweets(x))

        #call the function to analyze tweets
        df1['Sentiment']=df1['Text'].apply(lambda x: sentiment_analyze(x))
        
        # Overall Summary

        st.write("Total tweets extracted for twitter handle: {}: are:{}".format(Topic,len(df1['Text'])))
        st.write("Total Positive Tweets are:{}".format(len(df1[df1['Sentiment']=='positive😊'])))
        st.write("Total Neutral Tweets are:{}".format(len(df1[df1['Sentiment']=='neutral🙂'])))
        st.write("Total Negative Tweets are:{}".format(len(df1[df1['Sentiment']=='negative😑'])))

         # see the Extracted data
        if st.button("See the Extracted Data"):
            st.success("Below is the Extracted Data")
            st.write(df1.head(50))

        # get the count plot
        if st.button('Get Count Plot '):
            st.success("Generating a Count Plot")
            st.subheader("Count Plot for Different Sentiments")
            st.write(sns.countplot(x=df1['Sentiment']))
            st.pyplot()

        if st.button("Get Pie Chart"):
            st.success("Generating a Pie Chart")
            a=len(df1[df1['Sentiment']=='positive😊'])
            b=len(df1[df1['Sentiment']=='negative😑'])
            c=len(df1[df1['Sentiment']=='neutral🙂'])
            d=np.array([a,b,c])
            explode=(0.1,0.1,0.1)
            st.write(plt.pie(d,labels=['Positive','Negative','Neutral'],shadow=True,autopct='%1.2f%%',explode=explode))
            st.pyplot()

         # Create Wordcloud
        if st.button("Get Word Cloud"):
            st.success("Generating a Word Cloud")
            text=" ".join(review for review in df1['Clean Tweets'])
            stopwords=set(STOPWORDS)
            text_newALL=prepcloud(text,Topic)
            wordcloud=WordCloud(stopwords=stopwords,max_words=800,max_font_size=70).generate(text_newALL)
            st.write(plt.imshow(wordcloud,interpolation='bilinear'))
            plt.axis("off")
            st.pyplot()

        # WordCloud for Positive Tweets only
        if st.button("Get Word Cloud for all Positive Tweets"):
            st.success("Generating a WordCloud for all Positive Tweets")
            text=" ".join(review for review in df1[df1['Sentiment']=='positive😊']['Clean Tweets'])
            stopwords=set(STOPWORDS)
            text_newALL=prepcloud(text,Topic)
            wordcloud=WordCloud(stopwords=stopwords,max_words=800,max_font_size=70).generate(text_newALL)
            st.write(plt.imshow(wordcloud,interpolation='bilinear'))
            plt.axis("off")
            st.pyplot()

        # Wordcloud for all Neagtive Tweets
        if st.button("Get Word Cloud for all Negative Tweets "):
            st.success("Generating a WordCloud for all Negative Tweets")
            text=" ".join(review for review in df1[df1['Sentiment']=='negative😑']['Clean Tweets'])
            stopwords=set(STOPWORDS)
            text_newALL=prepcloud(text,Topic)
            wordcloud=WordCloud(stopwords=stopwords,max_words=800,max_font_size=70).generate(text_newALL)
            st.write(plt.imshow(wordcloud,interpolation='bilinear'))
            plt.axis("off")
            st.pyplot()        

if st.button("Exit"):
        st.balloons()

if __name__=='__main__':
    main()
            
            
            