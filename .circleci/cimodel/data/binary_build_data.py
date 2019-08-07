#!/usr/bin/env python3

"""
This module models the tree of configuration variants
for "smoketest" builds.

Each subclass of ConfigNode represents a layer of the configuration hierarchy.
These tree nodes encapsulate the logic for whether a branch of the hierarchy
should be "pruned".

In addition to generating config.yml content, the tree is also traversed
to produce a visualization of config dimensions.
"""

from collections import OrderedDict

from cimodel.lib.conf_tree import ConfigNode
import cimodel.data.dimensions as dimensions


LINKING_DIMENSIONS = [
    "shared",
    "static",
]


DEPS_INCLUSION_DIMENSIONS = [
    "with-deps",
    "without-deps",
]


def get_processor_arch_name(cuda_version):
    return "cpu" if not cuda_version else "cu" + cuda_version


LINUX_PACKAGE_VARIANTS = OrderedDict(
    manywheel=[
        "2.7m",
        "2.7mu",
        "3.5m",
        "3.6m",
        "3.7m",
    ],
    conda=dimensions.STANDARD_PYTHON_VERSIONS,
    libtorch=[
        "2.7m",
    ],
)

CONFIG_TREE_DATA = OrderedDict(
    linux=(dimensions.CUDA_VERSIONS, LINUX_PACKAGE_VARIANTS),
    macos=([None], OrderedDict(
        wheel=dimensions.STANDARD_PYTHON_VERSIONS,
        conda=dimensions.STANDARD_PYTHON_VERSIONS,
        libtorch=[
            "2.7",
        ],
    )),
)


# Why is this an option?
# All the nightlies used to be devtoolset3 and built with the old gcc ABI. We
# added a devtoolset7 option so that we could build nightlies with the new gcc
# ABI. That didn't work since devtoolset7 can't build with the new gcc ABI. But
# then we set devtoolset7 to be the default anyways, since devtoolset7
# understands avx512, which is needed for good fbgemm performance.
# This should be removed. The base dockers should just be upgraded to
# devtoolset7 so we don't have to reinstall this in every build job.
# The same machinery that this uses, though, should be retooled for a different
# compiler toolchain that can build with the new gcc ABI.
LINUX_GCC_CONFIG_VARIANTS = OrderedDict(
    manywheel=['devtoolset7'],
    conda=['devtoolset7'],
    libtorch=[
        "devtoolset7",
        "gcc5.4_cxx11",
    ],
)


class TopLevelNode(ConfigNode):
    def __init__(self, node_name, config_tree_data, smoke):
        super(TopLevelNode, self).__init__(None, node_name)

        self.config_tree_data = config_tree_data
        self.props["smoke"] = smoke

    def get_children(self):
        return [OSConfigNode(self, x, c, p) for (x, (c, p)) in self.config_tree_data.items()]


class OSConfigNode(ConfigNode):
    def __init__(self, parent, os_name, cuda_versions, py_tree):
        super(OSConfigNode, self).__init__(parent, os_name)

        self.py_tree = py_tree
        self.props["os_name"] = os_name
        self.props["cuda_versions"] = cuda_versions

    def get_children(self):
        packaging_variants = [PackageFormatConfigNode(self, k, v) for k, v in self.py_tree.items()]

        if self.find_prop("smoke"):
            filtered_packaging_variants = list(filter(lambda x: x.get_label() != "libtorch", packaging_variants))
            return filtered_packaging_variants
        else:
            return packaging_variants


class PackageFormatConfigNode(ConfigNode):
    def __init__(self, parent, package_format, python_versions):
        super(PackageFormatConfigNode, self).__init__(parent, package_format)

        self.props["python_versions"] = python_versions
        self.props["package_format"] = package_format

    def get_children(self):
        if self.find_prop("os_name") == "linux":
            return [LinuxGccConfigNode(self, v) for v in LINUX_GCC_CONFIG_VARIANTS[self.find_prop("package_format")]]
        else:
            return [ArchConfigNode(self, v) for v in self.find_prop("cuda_versions")]


class LinuxGccConfigNode(ConfigNode):
    def __init__(self, parent, devtoolset_version):
        super(LinuxGccConfigNode, self).__init__(parent, "DEVTOOLSET=" + str(devtoolset_version))

        self.props["devtoolset_version"] = devtoolset_version

    def get_children(self):
        cuda_versions = self.find_prop("cuda_versions")

        # XXX devtoolset7 on CUDA 9.0 is temporarily disabled
        # see https://github.com/pytorch/pytorch/issues/20066
        if self.find_prop("devtoolset_version") == 7:
            cuda_versions = filter(lambda x: x != "90", cuda_versions)

        return [ArchConfigNode(self, v) for v in cuda_versions]


class ArchConfigNode(ConfigNode):
    def __init__(self, parent, cu):
        super(ArchConfigNode, self).__init__(parent, get_processor_arch_name(cu))

        self.props["cu"] = cu

    def get_children(self):
        return [PyVersionConfigNode(self, v) for v in self.find_prop("python_versions")]


class PyVersionConfigNode(ConfigNode):
    def __init__(self, parent, pyver):
        super(PyVersionConfigNode, self).__init__(parent, pyver)

        self.props["pyver"] = pyver

    def get_children(self):

        smoke = self.find_prop("smoke")
        package_format = self.find_prop("package_format")
        os_name = self.find_prop("os_name")

        has_libtorch_variants = package_format == "libtorch" and os_name == "linux"
        linking_variants = LINKING_DIMENSIONS if has_libtorch_variants else []

        return [LinkingVariantConfigNode(self, v) for v in linking_variants]


class LinkingVariantConfigNode(ConfigNode):
    def __init__(self, parent, linking_variant):
        super(LinkingVariantConfigNode, self).__init__(parent, linking_variant)

    def get_children(self):
        return [DependencyInclusionConfigNode(self, v) for v in DEPS_INCLUSION_DIMENSIONS]


class DependencyInclusionConfigNode(ConfigNode):
    def __init__(self, parent, deps_variant):
        super(DependencyInclusionConfigNode, self).__init__(parent, deps_variant)

        self.props["libtorch_variant"] = "-".join([self.parent.get_label(), self.get_label()])
