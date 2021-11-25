import sys
sys.path.append('.')
import streamlit as st
from memenet.models import ImgNet, TxtNet

st.write('Hello, world! :)')
st.write(str(ImgNet))
st.write(str(TxtNet))
st.balloons()
