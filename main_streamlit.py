# simple_streamlit_app.py
"""
Una simple aplicación streamlit
ejecuta la aplicación instalando streamlit con pip y escribiendo
> streamlit ejecuta simple_streamlit_app.py
"""
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# Gensim
import gensim
# import nltk.data
import gensim.corpora as corpora
import streamlit.components.v1 as components

import gensim.models.callbacks
from wordcloud import WordCloud
from nltk.corpus import stopwords

from sklearn.manifold import TSNE
from bokeh.plotting import figure
from bokeh.io import output_notebook


print('StreamLit version', st.__version__)
print('Gensim version', gensim.__version__)


def process_words(texto, stop_word):
    """Convert a document into a list of lowercase tokens, ignoring tokens that are too short or too long.
    Uses :func:`~gensim.utils.tokenize` internally.
    """
    sentences = texto.split('\n')
    print(sentences)
    sentences = [gensim.utils.simple_preprocess(str(sent), deacc=True) for sent in sentences]

    # Filtrar stop words
    tokens = [[token for token in tokens if token not in stop_word] for tokens in sentences]
    print(tokens)
    return tokens


def create_model(data):
    print('Create Model', data)

    # Create Dictionary
    id2word = corpora.Dictionary(data)

    # Create Corpus: Term Document Frequency
    corpus = [id2word.doc2bow(text) for text in data]

    # Build LDA model
    lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus, id2word=id2word, random_state=100, update_every=1,
                                                chunksize=10, passes=10, alpha='symmetric', iterations=100,
                                                per_word_topics=True)
    return lda_model, corpus


def show_topis_models_lda(lda_model, corpus):
    # Get topic weights and dominant topics ------------
    # Get topic weights
    topic_weights = []
    for i, row_list in enumerate(lda_model[corpus]):
        topic_weights.append([w for i, w in row_list[0]])

    # Array of topic weights
    arr = pd.DataFrame(topic_weights).fillna(0).values

    # Keep the well separated points (optional)
    arr = arr[np.amax(arr, axis=1) > 0.35]

    # Dominant topic number in each doc
    topic_num = np.argmax(arr, axis=1)

    # tSNE Dimension Reduction
    tsne_model = TSNE(n_components=2, verbose=1, random_state=0, angle=.99, init='pca')
    tsne_lda = tsne_model.fit_transform(arr)

    # Plot the Topic Clusters using Bokeh
    output_notebook()
    n_topics = 4
    mycolors = np.array([color for name, color in mcolors.TABLEAU_COLORS.items()])
    plot = figure(title="t-SNE Clustering of {} LDA Topics".format(n_topics),
                  plot_width=900, plot_height=700)
    plot.scatter(x=tsne_lda[:, 0], y=tsne_lda[:, 1], color=mycolors[topic_num])
    # show(plot)
    st.bokeh_chart(plot)


def wordcloud_each_topic(lda_model, corpus):
    cols = [color for name, color in mcolors.TABLEAU_COLORS.items()]  # more colors: 'mcolors.XKCD_COLORS'

    cloud = WordCloud(background_color='white', width=2500, height=1800, max_words=10,
                      colormap='tab10', color_func=lambda *args, **kwargs: cols[i], prefer_horizontal=1.0)

    topics = lda_model.show_topics(formatted=False)

    fig, axes = plt.subplots(2, 2, figsize=(10,10), sharex=True, sharey=True)

    for i, ax in enumerate(axes.flatten()):
        fig.add_subplot(ax)
        topic_words = dict(topics[i][1])
        cloud.generate_from_frequencies(topic_words, max_font_size=300)
        plt.gca().imshow(cloud)
        plt.gca().set_title('Topic ' + str(i), fontdict=dict(size=16))
        plt.gca().axis('off')

    plt.subplots_adjust(wspace=0, hspace=0)
    plt.axis('off')
    plt.margins(x=0, y=0)
    plt.tight_layout()
    plt.show()
    st.pyplot(plt)


def get_stopwords(language):
    # print(nltk.data.path)
    return stopwords.words(language)


try:

    source_code_publi = """
    <script async src="https://pagead2.googlesyndication.com/pagead/js/adsbygoogle.js"></script>
    <!-- Test -->
    <ins class="adsbygoogle"
        style="display:block"
        data-ad-client="ca-pub-9843389507100478"
        data-ad-slot="9211203082"
        data-ad-format="auto"
        data-full-width-responsive="true"></ins>
    <script>
        (adsbygoogle = window.adsbygoogle || []).push({});
    </script>"""

    source_code = """
    <html>
      <head>
        <title>Agrupador de frases similares</title>
        <meta http-equiv = “Content-Type” content = “text / html; charset = utf-8”>
        <meta name = “description” content = “Aquí podrás agrupar diferentes frases, preguntas, comentarios, ...”>
        <meta name = “keywords” content = “agrupar, similar, frases, comentarios, preguntas”>
        <link rel = “stylesheet” type = “text / css” href = “/ style.css”>
        <link rel = “icono de acceso directo” href = “/ favicon.ico”>
      </head>

      <body>
      <h1>Agrupador de frases similares</h1>
      <div id = “mainContent”>
      <p>Bienvenido, aquí podrás agrupar diferentes frases, ya sean preguntas, comentarios, ...</p></div></body>
    </html>"""

    components.html(source_code)
    #st.title('Agrupador de frases similares')
    #st.markdown("Bienvenido, aquí podrás agrupar diferentes frases, ya sean preguntas, comentarios, ...")
    #st.header("Introduce el Texto")
    text = st.text_area('Escriba aquí las frases que desees agrupar')
    text = 'hola esto es una prueba\n prueba de estilo\nhola pepe como estas?'
    print(text)

    if text:

        stop_words = get_stopwords('spanish')
        print(stop_words)

        progress_bar = st.progress(0)
        status_text = st.empty()

        data_ready = process_words(text, stop_words)  # processed Text Data!
        status_text.text("%i%% Complete" % 25)
        progress_bar.progress(25)

        if len(data_ready) > 2:
            lda_model, corpus = create_model(data_ready)
            status_text.text("%i%% Complete" % 50)
            progress_bar.progress(50)

            #show_topis_models_lda(lda_model, corpus)
            wordcloud_each_topic(lda_model, corpus, None)
            status_text.text("%i%% Complete" % 100)
            progress_bar.progress(100)
            progress_bar.empty()

except Exception as error:
    print('ERROR:' + str(error))
    st.text('ERROR:', error)
    components.html(source_code_publi)
