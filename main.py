# simple_streamlit_app.py
"""
Una simple aplicación streamlit
ejecuta la aplicación instalando streamlit con pip y escribiendo
> streamlit ejecuta simple_streamlit_app.py
"""

import streamlit as st

st.title('Simple Streamlit App')
st.text('Tipo un número en el cuadro de abajo ')

n = st.number_input('Number')
st.write(f'{n} + 1 = {n + 1}')

s = st.text_input('Escriba a nombre en el cuadro de abajo ')
st.write(f'Hola {s}')
