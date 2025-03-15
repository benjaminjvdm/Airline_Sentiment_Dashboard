from io import BytesIO
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
from datetime import datetime
import requests

st.title("✈️ Sentiment Analysis of US Airlines Tweets Dashboard")
st.sidebar.title("Navigation & Filters")
st.markdown("""
    ### Interactive Tweet Analysis Dashboard
    Explore sentiment patterns of US airline tweets with interactive visualizations.
    """)
st.sidebar.markdown("Filter data and configure visualizations:")

# Define a function to cache the data to prevent reloading every time the app is run
@st.cache_data(persist=True)
def load_data():
    # Load the CSV file into a dataframe using Pandas' read_csv function
    data = pd.read_csv("Tweets.csv")
    # Convert 'tweet_created' column to datetime object
    data['tweet_created'] = pd.to_datetime(data['tweet_created'])
    # Create date column for filtering
    data['date'] = data['tweet_created'].dt.date
    return data

# Call the function to load the data
data = load_data()

# Add date range filter
min_date = data['date'].min()
max_date = data['date'].max()
selected_dates = st.sidebar.date_input(
    "Select date range",
    [min_date, max_date],
    min_value=min_date,
    max_value=max_date
)

# Filter data based on date selection
if len(selected_dates) == 2:
    data = data[(data['date'] >= selected_dates[0]) & (data['date'] <= selected_dates[1])]
else:
    st.sidebar.error("Please select a date range")

# Create tabs for better organization
tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Geospatial", "Airlines Analysis", "Text Analysis"])

# ========== Tab 1: Overview ==========
with tab1:
    st.header("General Statistics")
    
    col1, col2 = st.columns(2)
    with col1:
        # Create a widget to choose between a bar plot or pie chart
        st.subheader("Tweet Sentiment Distribution")
        viz_type = st.selectbox('Select visualization', ['Bar plot', 'Pie chart'])
        sentiment_count = data['airline_sentiment'].value_counts().reset_index()
        sentiment_count.columns = ['Sentiment', 'Count']
        
        if viz_type == 'Bar plot':
            fig = px.bar(sentiment_count, x='Sentiment', y='Count', 
                         color='Sentiment', height=400)
            st.plotly_chart(fig, use_container_width=True)
        else:
            fig = px.pie(sentiment_count, values='Count', names='Sentiment')
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Add temporal analysis
        st.subheader("Sentiment Over Time")
        time_agg = st.selectbox("Time aggregation", 
                              ['Hourly', 'Daily', 'Weekly'], 
                              key='time_agg')
        
        if time_agg == 'Hourly':
            data['time'] = data['tweet_created'].dt.hour
        elif time_agg == 'Daily':
            data['time'] = data['tweet_created'].dt.date
        else:
            data['time'] = data['tweet_created'].dt.to_period('W').astype(str)
            
        time_series = data.groupby(['time', 'airline_sentiment']).size().reset_index(name='count')
        fig = px.line(time_series, x='time', y='count', 
                      color='airline_sentiment', 
                      labels={'count': 'Number of Tweets'},
                      height=400)
        st.plotly_chart(fig, use_container_width=True)

# ========== Tab 2: Geospatial ==========
with tab2:
    st.header("Geospatial Analysis")
    
    # Hour selection with enhanced time range
    hour_range = st.slider("Select hour range", 0, 23, (9, 17))
    filtered_data = data[
        (data['tweet_created'].dt.hour >= hour_range[0]) & 
        (data['tweet_created'].dt.hour <= hour_range[1])
    ]
    
    st.subheader(f"Tweet Locations ({hour_range[0]}:00 - {hour_range[1]}:00)")
    st.markdown(f"**{len(filtered_data)} tweets found**")
    st.map(filtered_data)
    
    if st.checkbox("Show filtered data preview"):
        st.dataframe(filtered_data[['text', 'airline', 'airline_sentiment', 'tweet_created']])

# ========== Tab 3: Airlines Analysis ==========
with tab3:
    st.header("Airline-specific Analysis")
    
    # Airline comparison
    st.subheader("Airline Performance Comparison")
    selected_airlines = st.multiselect(
        "Select airlines to compare",
        options=data['airline'].unique(),
        default=['United', 'American', 'Delta']
    )
    
    if selected_airlines:
        # Create subplots
        fig = make_subplots(rows=1, cols=len(selected_airlines), 
                            subplot_titles=selected_airlines)
        
        for idx, airline in enumerate(selected_airlines):
            airline_data = data[data['airline'] == airline]
            counts = airline_data['airline_sentiment'].value_counts()
            
            trace = go.Bar(
                x=counts.index,
                y=counts.values,
                name=airline,
                text=counts.values,
                textposition='auto'
            )
            fig.add_trace(trace, row=1, col=idx+1)
        
        fig.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Please select at least one airline")

# ========== Tab 4: Text Analysis ==========
with tab4:
    st.header("Textual Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Enhanced word cloud settings
        st.subheader("Word Cloud Settings")
        word_sentiment = st.selectbox(
            'Select sentiment for word cloud',
            ('positive', 'neutral', 'negative')
        )
        max_words = st.slider("Maximum words", 50, 300, 150)
        colormap = st.selectbox(
            "Color theme",
            ['viridis', 'plasma', 'inferno', 'magma', 'cividis']
        )
    
    with col2:
        st.subheader("Generated Word Cloud")
        if word_sentiment:
            df = data[data['airline_sentiment'] == word_sentiment]
            text = ' '.join(tweet for tweet in df['text'])
            
            # Enhanced text cleaning
            stopwords = set(STOPWORDS)
            custom_stopwords = {'http', 'https', 'co', 'RT'}
            stopwords.update(custom_stopwords)
            
            wordcloud = WordCloud(
                stopwords=stopwords,
                max_words=max_words,
                colormap=colormap,
                background_color='white',
                width=800, 
                height=400
            ).generate(text)
            
            fig, ax = plt.subplots()
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.axis('off')
            st.pyplot(fig)
            
            # Add download button
            buf = BytesIO()
            plt.savefig(buf, format='png')
            st.download_button(
                label="Download Word Cloud",
                data=buf.getvalue(),
                file_name=f"{word_sentiment}_wordcloud.png",
                mime="image/png"
            )

# ========== Sidebar Additions ==========
st.sidebar.markdown("---")
st.sidebar.header("Data Export")
if st.sidebar.button("Download Full Dataset as CSV"):
    csv = data.to_csv(index=False).encode('utf-8')
    st.sidebar.download_button(
        label="Download CSV",
        data=csv,
        file_name="airline_tweets.csv",
        mime="text/csv"
    )

st.sidebar.markdown("---")
st.subheader("About the Author:")

    image_url = "https://github.com/benjaminjvdm/streamlit-author/blob/main/Untitled%20design(1)(1).png?raw=true"
    try:
        response = requests.get(image_url)
        response.raise_for_status()
        image = response.content
        st.image(image, caption="Moon Benjee (문벤지)", use_container_width=True)
    except requests.exceptions.RequestException as e:
        st.error(f"Error loading image: {e}")

    st.markdown(
        """
        This app was Built with ❤️ by **Benjee(문벤지)**. 
        You can connect with me on: [LinkedIn](https://www.linkedin.com/in/benjaminjvdm/)
        """
    )
st.sidebar.info("""
    **Dashboard Features:**
    - Interactive date and time filters
    - Comparative airline analysis
    - Temporal sentiment trends
    - Customizable word clouds
    - Data export capabilities
""")
