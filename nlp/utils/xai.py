"""
Utility functions for XAI.
"""
import torch
from captum.attr import LayerIntegratedGradients


def summarize_attributions(attributions):
    attributions = attributions.sum(dim=-1).squeeze(0)
    attributions = attributions / torch.norm(attributions)
    return attributions


def lig_attribute(forward_func, layer, input_ids, ref_input_ids, mask):
    lig = LayerIntegratedGradients(forward_func, layer)

    # compute attributions and approximation delta using layer integrated gradients
    attributions_ig, delta = lig.attribute(
        inputs=input_ids,
        baselines=ref_input_ids,
        additional_forward_args=(mask,),
        return_convergence_delta=True,
    )
    attributions = summarize_attributions(attributions_ig)

    values = attributions.detach().cpu().numpy()
    base_values = delta.detach().cpu().numpy()
    return values, base_values
