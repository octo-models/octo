from functools import partial
import itertools
from multiprocessing import Pool
from typing import Any, Callable, Dict, Iterable, Tuple, Union

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow_datasets.core import (
    dataset_builder,
    download,
    example_serializer,
    file_adapters,
    naming,
)
from tensorflow_datasets.core import split_builder as split_builder_lib
from tensorflow_datasets.core import splits as splits_lib
from tensorflow_datasets.core import utils
from tensorflow_datasets.core import writer as writer_lib

Key = Union[str, int]
# The nested example dict passed to `features.encode_example`
Example = Dict[str, Any]
KeyExample = Tuple[Key, Example]


class MultiThreadedAdhocDatasetBuilder(tfds.core.dataset_builders.AdhocBuilder):
    """Multithreaded adhoc dataset builder."""

    def __init__(
        self, *args, generator_fcn, n_workers, max_episodes_in_memory, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self._generator_fcn = generator_fcn
        self._n_workers = n_workers
        self._max_episodes_in_memory = max_episodes_in_memory

    def _download_and_prepare(  # pytype: disable=signature-mismatch  # overriding-parameter-type-checks
        self,
        dl_manager: download.DownloadManager,
        download_config: download.DownloadConfig,
    ) -> None:
        """Generate all splits and returns the computed split infos."""
        assert (
            self._max_episodes_in_memory % self._n_workers == 0
        )  # need to divide max_episodes by workers
        split_builder = ParallelSplitBuilder(
            split_dict=self._split_datasets,
            features=self.info.features,
            dataset_size=self.info.dataset_size,
            max_examples_per_split=download_config.max_examples_per_split,
            beam_options=download_config.beam_options,
            beam_runner=download_config.beam_runner,
            file_format=self.info.file_format,
            shard_config=download_config.get_shard_config(),
            generator_fcn=self._generator_fcn,
            n_workers=self._n_workers,
            max_episodes_in_memory=self._max_episodes_in_memory,
        )
        split_generators = self._split_generators(dl_manager)
        split_generators = split_builder.normalize_legacy_split_generators(
            split_generators=split_generators,
            generator_fn=self._generate_examples,
            is_beam=False,
        )
        dataset_builder._check_split_names(split_generators.keys())

        # Start generating data for all splits
        path_suffix = file_adapters.ADAPTER_FOR_FORMAT[
            self.info.file_format
        ].FILE_SUFFIX

        split_info_futures = []
        for split_name, generator in utils.tqdm(
            split_generators.items(),
            desc="Generating splits...",
            unit=" splits",
            leave=False,
        ):
            filename_template = naming.ShardedFileTemplate(
                split=split_name,
                dataset_name=self.name,
                data_dir=self.data_path,
                filetype_suffix=path_suffix,
            )
            future = split_builder.submit_split_generation(
                split_name=split_name,
                generator=generator,
                filename_template=filename_template,
                disable_shuffling=self.info.disable_shuffling,
            )
            split_info_futures.append(future)

        # Finalize the splits (after apache beam completed, if it was used)
        split_infos = [future.result() for future in split_info_futures]

        # Update the info object with the splits.
        split_dict = splits_lib.SplitDict(split_infos)
        self.info.set_splits(split_dict)

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Define dummy split generators."""

        def dummy_generator():
            yield None

        return {split: dummy_generator() for split in self._split_datasets}


class _SplitInfoFuture:
    """Future containing the `tfds.core.SplitInfo` result."""

    def __init__(self, callback: Callable[[], splits_lib.SplitInfo]):
        self._callback = callback

    def result(self) -> splits_lib.SplitInfo:
        return self._callback()


def parse_examples_from_generator(
    episodes, max_episodes, fcn, split_name, total_num_examples, features, serializer
):
    upper = episodes[-1] + 1
    upper_str = f'{upper}' if upper < max_episodes else ''
    generator = fcn(split=split_name + f"[{episodes[0]}:{upper_str}]")
    outputs = []
    for key, sample in utils.tqdm(
        zip(episodes, generator),
        desc=f"Generating {split_name} examples...",
        unit=" examples",
        total=total_num_examples,
        leave=False,
        mininterval=1.0,
    ):
        if sample is None:
            continue
        try:
            sample = features.encode_example(sample)
        except Exception as e:  # pylint: disable=broad-except
            utils.reraise(e, prefix=f"Failed to encode example:\n{sample}\n")
        outputs.append((str(key), serializer.serialize_example(sample)))
    return outputs


class ParallelSplitBuilder(split_builder_lib.SplitBuilder):
    def __init__(
        self, *args, generator_fcn, n_workers, max_episodes_in_memory, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self._generator_fcn = generator_fcn
        self._n_workers = n_workers
        self._max_episodes_in_memory = max_episodes_in_memory

    def _build_from_generator(
        self,
        split_name: str,
        generator: Iterable[KeyExample],
        filename_template: naming.ShardedFileTemplate,
        disable_shuffling: bool,
    ) -> _SplitInfoFuture:
        """Split generator for example generators.

        Args:
          split_name: str,
          generator: Iterable[KeyExample],
          filename_template: Template to format the filename for a shard.
          disable_shuffling: Specifies whether to shuffle the examples,

        Returns:
          future: The future containing the `tfds.core.SplitInfo`.
        """
        total_num_examples = None
        serialized_info = self._features.get_serialized_info()
        writer = writer_lib.Writer(
            serializer=example_serializer.ExampleSerializer(serialized_info),
            filename_template=filename_template,
            hash_salt=split_name,
            disable_shuffling=disable_shuffling,
            file_format=self._file_format,
            shard_config=self._shard_config,
        )

        del generator  # use parallel generators instead
        episode_lists = chunk_max(
            list(np.arange(self._split_dict[split_name].num_examples)),
            self._n_workers,
            self._max_episodes_in_memory,
        )  # generate N episode lists
        print(f"Generating with {self._n_workers} workers!")
        pool = Pool(processes=self._n_workers)
        for i, episodes in enumerate(episode_lists):
            print(f"Processing chunk {i + 1} of {len(episode_lists)}.")
            results = pool.map(
                partial(
                    parse_examples_from_generator,
                    fcn=self._generator_fcn,
                    split_name=split_name,
                    total_num_examples=total_num_examples,
                    serializer=writer._serializer,
                    features=self._features,
                    max_episodes=self._split_dict[split_name].num_examples,
                ),
                episodes,
            )
            # write results to shuffler --> this will automatically offload to disk if necessary
            print("Writing conversion results...")
            for result in itertools.chain(*results):
                key, serialized_example = result
                writer._shuffler.add(key, serialized_example)
                writer._num_examples += 1
        pool.close()

        print("Finishing split conversion...")
        shard_lengths, total_size = writer.finalize()

        split_info = splits_lib.SplitInfo(
            name=split_name,
            shard_lengths=shard_lengths,
            num_bytes=total_size,
            filename_template=filename_template,
        )
        return _SplitInfoFuture(lambda: split_info)


def dictlist2listdict(DL):
    "Converts a dict of lists to a list of dicts"
    return [dict(zip(DL, t)) for t in zip(*DL.values())]


def chunks(l, n):
    """Yield n number of sequential chunks from l."""
    d, r = divmod(len(l), n)
    for i in range(n):
        si = (d + 1) * (i if i < r else r) + d * (0 if i < r else i - r)
        yield l[si : si + (d + 1 if i < r else d)]


def chunk_max(l, n, max_chunk_sum):
    out = []
    for _ in range(int(np.ceil(len(l) / max_chunk_sum))):
        out.append([c for c in chunks(l[:max_chunk_sum], n) if c])
        l = l[max_chunk_sum:]
    return out
