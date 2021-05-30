"""
Utility functions for plots.
"""
import random
import string

import numpy as np
from shap.plots import colors
from shap.plots._text import unpack_shap_explanation_contents, process_shap_values


class SV:
    def __init__(self, values, base_values, data):
        self.values = np.array(values)
        self.base_values = base_values
        self.data = data


def visualize_text(values, base_values, data):
    """Wrapper for text.""" 
    return text(SV(values, base_values, data))


def text(shap_values, num_starting_labels=0, group_threshold=1, separator='', xmin=None, xmax=None, cmax=None):
    """Visualize text.
    Adapted from SHAP shap.plots._text
    """
    def values_min_max(values, base_values):
        """ Used to pick our axis limits.
        """
        fx = base_values + values.sum()
        xmin = fx - values[values > 0].sum()
        xmax = fx - values[values < 0].sum()
        cmax = max(abs(values.min()), abs(values.max()))
        d = xmax - xmin
        xmin -= 0.1 * d
        xmax += 0.1 * d

        return xmin, xmax, cmax

    xmin_new, xmax_new, cmax_new = values_min_max(shap_values.values, shap_values.base_values)
    if xmin is None:
        xmin = xmin_new
    if xmax is None:
        xmax = xmax_new
    if cmax is None:
        cmax = cmax_new
    
    values, clustering = unpack_shap_explanation_contents(shap_values)
    tokens, values, group_sizes = process_shap_values(shap_values.data, values, group_threshold, separator, clustering)

    # build out HTML output one word one at a time
    top_inds = np.argsort(-np.abs(values))[:num_starting_labels]

    uuid = "".join(random.choices(string.ascii_lowercase, k=20))

    out = ""
    for i in range(len(tokens)):
        scaled_value = 0.5 + 0.5 * values[i] / cmax
        color = colors.red_transparent_blue(scaled_value)
        color = ((1-color[0])*255, (1-color[1])*255, color[2]*255, color[3])
        
        # display the labels for the most important words
        label_display = "none"
        wrapper_display = "inline"
        if i in top_inds:
            label_display = "block"
            wrapper_display = "inline-block"
        
        # create the value_label string
        value_label = ""
        if group_sizes[i] == 1:
            value_label = str(values[i].round(3))
        else:
            value_label = str(values[i].round(3)) + " / " + str(group_sizes[i])
        
        # the HTML for this token
        out += "<div style='display: " + wrapper_display + "; text-align: center;'>" \
             + "<div style='display: " + label_display + "; color: #999; padding-top: 0px; font-size: 12px;'>" \
             + value_label \
             + "</div>" \
             + f"<div id='_tp_{uuid}_ind_{i}'" \
             +   "style='display: inline; background: rgba" + str(color) + "; border-radius: 3px; padding: 0px'" \
             +   "onclick=\"if (this.previousSibling.style.display == 'none') {" \
             +       "this.previousSibling.style.display = 'block';" \
             +       "this.parentNode.style.display = 'inline-block';" \
             +     "} else {" \
             +       "this.previousSibling.style.display = 'none';" \
             +       "this.parentNode.style.display = 'inline';" \
             +     "}" \
             +   "\"" \
             +   f"onmouseover=\"document.getElementById('_fb_{uuid}_ind_{i}').style.opacity = 1; document.getElementById('_fs_{uuid}_ind_{i}').style.opacity = 1;" \
             +   "\"" \
             +   f"onmouseout=\"document.getElementById('_fb_{uuid}_ind_{i}').style.opacity = 0; document.getElementById('_fs_{uuid}_ind_{i}').style.opacity = 0;" \
             +   "\"" \
             + ">" \
             + tokens[i].replace("<", "&lt;").replace(">", "&gt;").replace(' ##', '') \
             + "</div>" \
             + "</div>"
    return out
