from pathlib import Path
from unittest import TestCase
from xml.etree import ElementTree

import markdown


class LazyImageExtensionTests(TestCase):
    def test_lead_image_is_prioritized_and_later_images_are_lazy(self):
        try:
            from hooks.lazy_images import LazyImagesExtension
        except ModuleNotFoundError:
            self.fail("hooks.lazy_images is not implemented")

        html = markdown.markdown(
            "![lead](lead.png)\n\n![detail](detail.png)",
            extensions=[LazyImagesExtension()],
        )
        root = ElementTree.fromstring(f"<root>{html}</root>")
        lead, detail = root.findall(".//img")

        self.assertEqual(lead.attrib["decoding"], "async")
        self.assertEqual(lead.attrib["fetchpriority"], "high")
        self.assertNotIn("loading", lead.attrib)
        self.assertEqual(detail.attrib["decoding"], "async")
        self.assertEqual(detail.attrib["loading"], "lazy")

    def test_explicit_image_loading_attributes_are_preserved(self):
        try:
            from hooks.lazy_images import LazyImagesExtension
        except ModuleNotFoundError:
            self.fail("hooks.lazy_images is not implemented")

        html = markdown.markdown(
            '![image](image.png){ loading="eager" decoding="sync" }',
            extensions=["attr_list", LazyImagesExtension()],
        )
        root = ElementTree.fromstring(f"<root>{html}</root>")
        image = root.find(".//img")

        self.assertIsNotNone(image)
        self.assertEqual(image.attrib["loading"], "eager")
        self.assertEqual(image.attrib["decoding"], "sync")

    def test_mkdocs_hook_registers_the_markdown_extension(self):
        from hooks import lazy_images

        config = {"markdown_extensions": []}
        lazy_images.on_config(config)

        self.assertIsInstance(config["markdown_extensions"][0], lazy_images.LazyImagesExtension)

    def test_explicit_high_priority_image_is_not_made_lazy(self):
        from hooks.lazy_images import LazyImagesExtension

        html = markdown.markdown(
            '![lead](lead.png)\n\n![priority](priority.png){ fetchpriority="high" }',
            extensions=["attr_list", LazyImagesExtension()],
        )
        root = ElementTree.fromstring(f"<root>{html}</root>")
        _, priority = root.findall(".//img")

        self.assertEqual(priority.attrib["fetchpriority"], "high")
        self.assertNotIn("loading", priority.attrib)
