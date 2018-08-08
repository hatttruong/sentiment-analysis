"""Summary

Attributes:
    logger (TYPE): Description
"""
import matplotlib as mpl
# mpl.use('Agg')
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.manifold import MDS
from sklearn.metrics.pairwise import cosine_similarity
import mpld3
import numpy as np
import logging


logger = logging.getLogger(__name__)


class TopToolbar(mpld3.plugins.PluginBase):
    """Plugin for moving toolbar to top of figure

    Attributes:
        dict_ (dict): Description
        JAVASCRIPT (TYPE): Description
    """

    JAVASCRIPT = """
    mpld3.register_plugin("toptoolbar", TopToolbar);
    TopToolbar.prototype = Object.create(mpld3.Plugin.prototype);
    TopToolbar.prototype.constructor = TopToolbar;
    function TopToolbar(fig, props){
        mpld3.Plugin.call(this, fig, props);
    };

    TopToolbar.prototype.draw = function(){
      // the toolbar svg doesn't exist
      // yet, so first draw it
      this.fig.toolbar.draw();

      // then change the y position to be
      // at the top of the figure
      this.fig.toolbar.toolbar.attr("x", 150);
      this.fig.toolbar.toolbar.attr("y", 400);

      // then remove the draw function,
      // so that it is not called again
      this.fig.toolbar.draw = function() {}
    }
    """

    def __init__(self):
        """Summary
        """
        self.dict_ = {"type": "toptoolbar"}


def visualize_word_with_tooltip(xs, ys, words, file_name):
    """
    define custom css to format the font and to remove the axis labeling


    Args:
        xs (TYPE): Description
        ys (TYPE): Description
        words (TYPE): Description
        file_name (TYPE): Description

    Deleted Parameters:
        true_labels (TYPE): Description
        titles (TYPE): Description
    """
    logger.info('start to visualize word with tooltip')
    css = """
    text.mpld3-text, div.mpld3-tooltip {
      font-family:Arial, Helvetica, sans-serif;
    }

    g.mpld3-xaxis, g.mpld3-yaxis {
    display: none; }

    svg.mpld3-figure {
    margin-left: -200px;}
    """

    # Plot
    fig, ax = plt.subplots(figsize=(14, 6))  # set plot size
    ax.margins(0.03)  # Optional, just adds 5% padding to the autoscaling

    points = ax.plot(xs, ys, marker='o', linestyle='', ms=18,
                     label=words, mec='none',
                     color='#1b9e77')
    ax.set_aspect('auto')

    # set tooltip using points, labels and the already defined 'css'
    tooltip = mpld3.plugins.PointHTMLTooltip(
        points[0], words, voffset=10, hoffset=10, css=css)

    # connect tooltip to fig
    mpld3.plugins.connect(fig, tooltip, TopToolbar())

    # set tick marks as blank
    ax.axes.get_xaxis().set_ticks([])
    ax.axes.get_yaxis().set_ticks([])

    # set axis as blank
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)

    # ax.legend(numpoints=1)  # show legend with only one dot

    # uncomment the below to export to html
    html = mpld3.fig_to_html(fig)
    with open(file_name, 'w') as f:
        f.write(html)


def compute_dist(matrix_data):
    """Summary

    Args:
        matrix_data (TYPE): (n_samples, n_features)

    Returns:
        TYPE: Description
    """
    logger.info('compute distance from data')
    return 1 - cosine_similarity(matrix_data)


def dimension_reduction_mds(dist):
    """
    Convert two components as we're plotting points in a two-dimensional plane
    "precomputed" because we provide a distance matrix
    we will also specify `random_state` so the plot is reproducible.

    Args:
        dist (TYPE): Description

    Returns:
        TYPE: Description
    """
    logger.info('start to dimension reduction using mds algorithm')

    MDS()
    mds = MDS(n_components=2, dissimilarity="precomputed", random_state=1)
    pos = mds.fit_transform(dist)  # shape (n_components, n_samples)
    xs, ys = pos[:, 0], pos[:, 1]
    return xs, ys


def mds_plot(wv_model, path, nb_words=None):
    """Summary

    Args:
        wv_model (TYPE): Description
        path (TYPE): Description
    """
    size = 0
    words = []
    word_vectors = []
    for word in wv_model.vocab:
        word_vectors.append(np.array(wv_model[word]))
        words.append(word)
        size += 1
        if nb_words is not None and size >= nb_words:
            break

    matrix_data = np.array(word_vectors)
    logger.info('matrix_data.shape: %s', matrix_data.shape)
    dist = compute_dist(matrix_data)
    xs, ys = dimension_reduction_mds(dist)
    visualize_word_with_tooltip(xs, ys, words, path)


def tsne_plot(model_wv, output_path=None):
    """
    Creates and TSNE model and plots it

    Args:
        model_wv (TYPE): Description
        output_path (None, optional): Description
    """
    labels = []
    tokens = []

    for word in model_wv.vocab:
        tokens.append(model_wv[word])
        labels.append(word)

    tsne_model = TSNE(perplexity=40, n_components=2,
                      init='pca', n_iter=2500, random_state=23)
    new_values = tsne_model.fit_transform(tokens)

    x = []
    y = []
    for value in new_values:
        x.append(value[0])
        y.append(value[1])

    visualize_word_with_tooltip(x, y, labels, output_path)
