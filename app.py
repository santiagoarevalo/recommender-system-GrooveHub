import streamlit as st
import pandas as pd
import joblib

st.title('Deployment of Clustering Model for GrooveHub')

st.write('This is the deployment of the clustering model for GrooveHub. The model is trained on a Spotify attributes tracks dataset and is used to cluster the songs into 5 groups based on their attributes. The model is trained on the following features:')

st.write('1. Danceability')
st.write('2. Speechiness')


def user_input_features():
    danceability = st.slider('Danceability', 0.0, 1.0, 0.5)
    speechiness = st.slider('Speechiness', 0.0, 1.0, 0.5)
    data = {'danceability': danceability,
            'speechiness': speechiness}
    features = pd.DataFrame(data, index=[0])
    return features

def cluster_0():
    st.write('The song belongs to the cluster of songs with low speechiness and medium danceability')

def cluster_1():
    st.write('The song belongs to the cluster of songs with very high speechiness and high danceability')

def cluster_2():
    st.write('The song belongs to the cluster of songs with low speechiness and high danceability')

def cluster_3():
    st.write('The song belongs to the cluster of songs with low speechiness and low danceability')

def cluster_4():
    st.write('The song belongs to the cluster of songs with medium speechiness and medium to high danceability')


def main():
    df = user_input_features()

    st.subheader('User Input parameters')
    st.write(df)

    model = joblib.load('kmeans.joblib')

    if st.button('Predict'):
        prediction = model.predict(df)
        st.write('The song belongs to cluster {}'.format(prediction[0]))
        switcher = {
            0: cluster_0,
            1: cluster_1,
            2: cluster_2,
            3: cluster_3,
            4: cluster_4
        }
        switcher.get(prediction[0])()

if __name__ == '__main__':
    main()
