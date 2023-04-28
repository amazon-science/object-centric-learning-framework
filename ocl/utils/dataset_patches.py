"""Patches `torchdata` for behavior to be consistent with webdatasets."""
import re
from typing import Callable, Dict

import torchdata
from torchdata.datapipes.iter import IterDataPipe


def patched_pathsplit(p):
    """Split a path into a WebDataset prefix and suffix.

    The version of pathsplit in torchdata behaves
    differently from WebDatasets by keeping "." in the
    suffix. This is patched here, by excluding the
    separating dot from the regex match.


    The prefix is used for grouping files into samples,
    the suffix is used as key in the output dictionary.
    The suffix consists of all components after the first
    "." in the filename.

    In torchdata, the prefix consists of the .tar file
    path followed by the file name inside the archive.

    Any backslash in the prefix is replaced by a forward
    slash to make Windows prefixes consistent with POSIX
    paths.
    """
    # convert Windows pathnames to UNIX pathnames, otherwise
    # we get an inconsistent mix of the Windows path to the tar
    # file followed by the POSIX path inside that tar file
    p = p.replace("\\", "/")
    if "." not in p:
        return p, ""
    # we need to use a regular expression because os.path is
    # platform specific, but tar files always contain POSIX paths
    # Patched here, exclude the extension dot from the regex expression.
    match = re.search(r"^.*/(.*?)\.([^/]*)$", p)
    pass
    if not match:
        return p, ""
    prefix, suffix = match.groups()
    return prefix, suffix


# Patch pathsplit implementation in torchdata as it does not follow webdataset
# exactly.
torchdata.datapipes.iter.util.webdataset.pathsplit = patched_pathsplit


@torchdata.datapipes.functional_datapipe("map_dict")
class DictMapper(IterDataPipe):
    def __init__(self, source_datapipe: IterDataPipe, mapping: Dict[str, Callable]):
        self.source_datapipe = source_datapipe
        self.mapping = mapping

    def __iter__(self):
        for dictionary in self.source_datapipe:
            output_dict = dictionary.copy()
            for name, fn in self.mapping.items():
                output_dict[name] = fn(dictionary[name])
            yield output_dict


@torchdata.datapipes.functional_datapipe("then")
class ChainedGenerator(IterDataPipe):
    """Simple interface to allow chaining via a generator function.

    This mirrors functionality from the webdatasets package.
    """

    def __init__(self, source_datapipe: IterDataPipe, generator):
        self.source_datapipe = source_datapipe
        self.generator = generator

    def __iter__(self):
        yield from self.generator(self.source_datapipe)
