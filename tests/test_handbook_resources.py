"""Catch broken article resources using the site's Markdown anchor generation."""
from functools import lru_cache
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit

import markdown
import pathspec
import yaml

DOCS = Path(__file__).resolve().parents[1] / 'files'


class Resources(HTMLParser):
    def __init__(self, html):
        super().__init__()
        self.ids, self.targets = set(), []
        self.feed(html)

    def handle_starttag(self, tag, attributes):
        attrs = dict(attributes)
        if 'id' in attrs:
            self.ids.add(attrs['id'])
        target = attrs.get('href') if tag == 'a' else attrs.get('src') if tag == 'img' else None
        if target:
            self.targets.append(target)


@lru_cache(maxsize=None)
def resources(page):
    return Resources(markdown.markdown(
        page.read_text(encoding='utf-8'),
        extensions=['toc', 'attr_list', 'tables', 'md_in_html', 'pymdownx.superfences',
                    'pymdownx.arithmatex'],
        extension_configs={'pymdownx.arithmatex': {'generic': True}},
    ))


def test_nonpaper_article_links_images_and_fragments_resolve():
    issues = []
    config = yaml.load((DOCS.parent / 'mkdocs.yml').read_text(), Loader=yaml.BaseLoader)
    excluded = pathspec.GitIgnoreSpec.from_lines(config.get('exclude_docs', '').splitlines())
    for page in DOCS.rglob('*.md'):
        if page.relative_to(DOCS).parts[0] in {'assets', 'paper-deep-dives'}:
            continue
        for href in resources(page).targets:
            url = urlsplit(href)
            if url.scheme or url.netloc or not (url.path or url.fragment):
                continue
            target = (DOCS / unquote(url.path).lstrip('/')) if url.path.startswith('/') else (
                page.parent / unquote(url.path) if url.path else page)
            target = target.resolve()
            if target.is_dir():
                target /= 'index.md'
            if not target.is_file():
                issues.append(f'{page.relative_to(DOCS)} -> {href}: missing file')
            elif target.is_relative_to(DOCS) and excluded.match_file(target.relative_to(DOCS).as_posix()):
                issues.append(f'{page.relative_to(DOCS)} -> {href}: excluded from built site')
            elif url.fragment and target.suffix == '.md' and unquote(url.fragment) not in resources(target).ids:
                issues.append(f'{page.relative_to(DOCS)} -> {href}: missing section')
    assert not issues, '\n'.join(issues)
