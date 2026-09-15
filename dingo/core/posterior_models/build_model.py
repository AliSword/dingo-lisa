import torch

from dingo.core.posterior_models.normalizing_flow import NormalizingFlowPosteriorModel
from dingo.core.posterior_models.flow_matching import FlowMatchingPosteriorModel
from dingo.core.posterior_models.score_matching import ScoreDiffusionPosteriorModel
from dingo.core.posterior_models.trig_flow_matching import TrigFlowMatchingPosteriorModel
from dingo.core.utils.backward_compatibility import update_model_config


def build_model_from_kwargs(filename: str = None, settings: dict = None, **kwargs):
    """
    Returns a PosteriorModel based on a saved network or settings dict.

    The function is careful to choose the appropriate PosteriorModel class (e.g.,
    for a normalizing flow, flow matching, or score matching).

    Parameters
    ----------
    filename: str
        Path to a saved network (.pt).
    settings: dict
        Settings dictionary.
    kwargs
        Arguments forwarded to the model constructor.

    Returns
    -------
    PosteriorModel
    """
    if (filename is None) == (settings is None):
        raise ValueError(
            "Either a filename or a settings dict must be provided, but not both."
        )

    models_dict = {
        "normalizing_flow": NormalizingFlowPosteriorModel,
        "flow_matching": FlowMatchingPosteriorModel,
        "score_matching": ScoreDiffusionPosteriorModel,
        "trig_flow_matching": TrigFlowMatchingPosteriorModel,
    }

    if filename is not None:
        d = torch.load(filename, map_location="meta")
        update_model_config(d["metadata"]["train_settings"]["model"])  # Backward compat
        posterior_model_type = d["metadata"]["train_settings"]["model"][
            "posterior_model_type"
        ]
    else:
        update_model_config(settings["train_settings"]["model"])  # Backward compat
        posterior_model_type = settings["train_settings"]["model"][
            "posterior_model_type"
        ]

    if not posterior_model_type.lower() in models_dict:
        raise ValueError("No valid posterior model type specified.")

    model = models_dict[posterior_model_type.lower()]

    return model(model_filename=filename, metadata=settings, **kwargs)


def autocomplete_model_kwargs(model_kwargs: dict, data_sample: list):
    """
    Autocomplete the model kwargs from train_settings and data_sample from the dataloader:

    * set input dimension of embedding net to shape of data_sample[1]
    * set dimension of parameter space to len(data_sample[0])
    * set added_context flag of embedding net if required for gnpe proxies
    * set context dim of posterior model to output dim of embedding net + gnpe proxy dim

    Parameters
    ----------
    model_kwargs: dict
        Model settings, which are modified in-place.
    data_sample: list
        Sample from dataloader (e.g., wfd[0]) used for autocomplection.
        Should be of format [parameters, GW data, gnpe_proxies], where the
        last element is only there is GNPE proxies are required.
    """

    # set input dims from ifo_list and domain information
    model_kwargs["embedding_kwargs"]["input_dims"] = list(data_sample[1].shape)
    # set dimension of parameter space of posterior model
    model_kwargs["posterior_kwargs"]["input_dim"] = len(data_sample[0])
    # set added_context flag of embedding net if GNPE proxies are required
    # set context dim of nsf to output dim of embedding net + GNPE proxy dim
    try:
        gnpe_proxy_dim = len(data_sample[2])
        model_kwargs["embedding_kwargs"]["added_context"] = True
        model_kwargs["posterior_kwargs"]["context_dim"] = (
            model_kwargs["embedding_kwargs"]["output_dim"] + gnpe_proxy_dim
        )
    except IndexError:
        # data_sample has only [parameters, GW data] -- no GNPE proxies.
        model_kwargs["embedding_kwargs"]["added_context"] = False
        model_kwargs["posterior_kwargs"]["context_dim"] = model_kwargs[
            "embedding_kwargs"
        ]["output_dim"]
    except TypeError:
        # data_sample[2] exists but has no len() -- this is not a GNPE proxy
        # array, it's a plain scalar. This happens when SNR reweighting is
        # enabled (see AttachImportanceWeight / set_train_transforms) and no
        # GNPE context_parameters are configured: selected_keys becomes
        # ["inference_parameters", "waveform", "weight"], so data_sample[2]
        # is the per-sample importance weight (a float), not GNPE context.
        # train_epoch/test_epoch (base_model.py) always treat the LAST
        # element of data[1:] as the weight regardless of how many context
        # elements precede it, so the correct behavior here is the same as
        # the no-GNPE-proxies case above.
        model_kwargs["embedding_kwargs"]["added_context"] = False
        model_kwargs["posterior_kwargs"]["context_dim"] = model_kwargs[
            "embedding_kwargs"
        ]["output_dim"]
