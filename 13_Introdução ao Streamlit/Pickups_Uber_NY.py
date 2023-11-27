import time
import streamlit as st
import pandas as pd
import numpy as np
import streamlit.components.v1 as components

st.title('Tafera_Uber')

with st.spinner(text='In progress'):
    time.sleep(3)
    st.success('Done')

st.write('## Uber pickups in NYC')

DATE_COLUMN = 'date/time'
DATA_URL = ('https://s3-us-west-2.amazonaws.com/'
            'streamlit-demo-data/uber-raw-data-sep14.csv.gz')

@st.cache_data
def load_data(nrows):
    data = pd.read_csv(DATA_URL, nrows=nrows)
    lowercase = lambda x: str(x).lower()
    data.rename(lowercase, axis='columns', inplace=True)
    data[DATE_COLUMN] = pd.to_datetime(data[DATE_COLUMN])
    return data

data_load_state = st.text('Loading data...')
data = load_data(10000)
data_load_state.text("Done! (using st.cache_data)")

if st.checkbox('Show raw data'):
    st.subheader('Raw data')
    st.write(data)

st.subheader('Number of pickups by hour')
hist_values = np.histogram(data[DATE_COLUMN].dt.hour, bins=24, range=(0,24))[0]
st.bar_chart(hist_values)

# Some number in the range 0-23
hour_to_filter = st.slider('hour', 0, 23, 17)
filtered_data = data[data[DATE_COLUMN].dt.hour == hour_to_filter]

st.subheader('Map of all pickups at %s:00' % hour_to_filter)
st.map(filtered_data)

st.write('## Previsão do tempo para o Rio de Janeiro')
components.html(
    """
    <div id="cont_b7d982a1b7f7b7908ed840ed97ec0f3e"><script type="text/javascript" async src="https://www.tempo.com/wid_loader/b7d982a1b7f7b7908ed840ed97ec0f3e"></script></div>
    """,
    height=250,
)