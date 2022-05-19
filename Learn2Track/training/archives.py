# -*- coding: utf-8 -*-

# Previous methods added in the trainer:
import logging
import numpy as np

from dwi_ml.training.batch_loaders import BatchLoaderOneInput
from dwi_ml.training.batch_samplers import DWIMLBatchSampler
from torch.utils.data import DataLoader


def _compute_stats_on_a_few_batches(batch_sampler, batch_loader):
    """
    Since the exact data weight per batch can vary based on data
    augmentation in the batch sampler, we approximate the epoch stats
    using a sample batch.
    """
    # Use temporary RNG states to preserve random "coherency"
    # e.g. when resuming an experiment
    sampler_rng_state_bk = batch_sampler.np_rng.get_state()

    dataloader = DataLoader(batch_sampler.dataset,
                            batch_sampler=batch_sampler,
                            num_workers=0,
                            collate_fn=batch_loader.load_batch)

    # Get a sample batch to compute stats
    # Note that using this does not really work the same way as during
    # training. The __iter__ function of the batch sampler is called
    # 5 times, instead of "yielding" 5 times. So the whole memory of
    # streamlines and subjects that have already been used is resettled
    # each time, and there is a possibility that the same streamlines will
    # be sampled more than once. But this is just for stats so ok.
    logging.info("Running the dataloader for 5 iterations, just to "
                 "compute statistics..")
    sample_batches = [next(iter(dataloader))[0] for _ in range(5)]

    # Restore RNG states. toDo OK??? Voir avec Philippe
    batch_sampler.np_rng.set_state(sampler_rng_state_bk)

    if batch_sampler.dataset.is_lazy:
        batch_sampler.dataset.volume_cache_manager = None

    # Compute stats about epoch
    logging.info("Batch sampler has been ")
    batches_nb_points = []
    batches_nb_streamlines = []
    for sample_data in sample_batches:
        batches_nb_streamlines.append(len(sample_data))
        batches_nb_points.append(sum([len(s) for s in sample_data]))

    avg_batch_size_nb_streamlines = int(np.mean(batches_nb_streamlines))
    avg_batch_size_nb_points = int(np.mean(batches_nb_points))
    if avg_batch_size_nb_points == 0:
        raise ValueError("The allowed batch size ({}) is too small! "
                         "Sampling 0 streamlines per batch."
                         .format(batch_sampler.batch_size))

    logging.info("We have computed that in average, each batch has a "
                 "size of ~{} streamlines for a total of ~{} number of "
                 "datapoints)"
                 .format(avg_batch_size_nb_streamlines,
                         avg_batch_size_nb_points))

    return avg_batch_size_nb_streamlines, avg_batch_size_nb_points


def _estimate_nb_batches_per_epoch(max_batches_per_epochs,
                                   batch_sampler: DWIMLBatchSampler,
                                   batch_loader: BatchLoaderOneInput):
    """
    Compute the number of batches necessary to use all the available data
    for an epoch (but limiting this to max_nb_batches).

    Returns
    -------
    n_batches : int
        Approximate number of updates per epoch
    batch_size : int
        Batch size or approximate batch size.
    """
    dataset_size = batch_sampler.dataset.total_nb_streamlines[
        batch_sampler.streamline_group_idx]

    # Let's find out how many streamlines in average are sampled.
    if batch_sampler.batch_size_units == 'nb_streamlines':
        # This is straightforward
        batch_size = batch_sampler.batch_size
    else:  # batch_sampler.batch_size_units == 'length_mm':
        # We don't know the actual size in number of streamlines.
        batch_size, _ = _compute_stats_on_a_few_batches(batch_sampler,
                                                        batch_loader)

    # toDo
    #  None of these cases ensure us a fixed number of input points. If
    #  streamlines are compressed, there isn't much more we can do. If
    #  streamlines have been resampled during loading, though, we can
    #  approximate the actual batch size in number of points after data
    #  augmentation (ignored 2nd returned arg above). But how would we know
    #  how many batches are needed per epoch?

    # Define the number of batches per epoch
    n_batches = int(dataset_size / batch_size)
    n_batches_capped = min(n_batches, max_batches_per_epochs)

    logging.info("Dataset had {} streamlines (before data augmentation) "
                 "and each batch contains ~{} streamlines.\nWe will be "
                 "using approximately {} batches per epoch (but not more "
                 "than the allowed {}).\n"
                 .format(dataset_size, batch_size, n_batches,
                         max_batches_per_epochs))

    return n_batches_capped, batch_size
