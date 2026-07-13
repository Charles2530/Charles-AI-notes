"""Add browser-native loading hints to Markdown images."""

from __future__ import annotations

from markdown.extensions import Extension
from markdown.treeprocessors import Treeprocessor


class LazyImageTreeprocessor(Treeprocessor):
    def run(self, root):
        images = list(root.iter("img"))
        for index, image in enumerate(images):
            image.attrib.setdefault("decoding", "async")
            if index == 0:
                if image.attrib.get("loading") != "lazy":
                    image.attrib.setdefault("fetchpriority", "high")
            elif image.attrib.get("fetchpriority") != "high":
                image.attrib.setdefault("loading", "lazy")
        return root


class LazyImagesExtension(Extension):
    def extendMarkdown(self, md):
        md.treeprocessors.register(LazyImageTreeprocessor(md), "lazy_images", 4)


def makeExtension(**kwargs):
    return LazyImagesExtension(**kwargs)


def on_config(config):
    extensions = config["markdown_extensions"]
    if not any(isinstance(extension, LazyImagesExtension) for extension in extensions):
        extensions.append(LazyImagesExtension())
    return config
