"""Point README.md's relative links and images at GitHub before a PyPI build.

PyPI renders the README as the project page but has no copy of docs/ or the
images, so every relative link there would 404. Run in the release workflow
only (the README in the repo keeps its relative links):

    python .github/scripts/pypi_readme.py README.md v0.1.0

Links go to github.com/<repo>/blob|tree/<ref>/..., images to
raw.githubusercontent.com/<repo>/<ref>/..., both pinned to the release tag so
the PyPI page matches the released code.
"""
import os
import re
import sys

REPO = "connorwhite-online/loadpath"


def absolute(target, ref, image):
    if re.match(r"^([a-z][a-z0-9+.-]*:|#|/)", target, re.I):
        return target   # already absolute, a mailto:, or an in-page anchor
    path, _, anchor = target.partition("#")
    path = path[2:] if path.startswith("./") else path
    if image:
        url = f"https://raw.githubusercontent.com/{REPO}/{ref}/{path}"
    else:
        kind = "tree" if path.endswith("/") or os.path.isdir(path) else "blob"
        url = f"https://github.com/{REPO}/{kind}/{ref}/{path}"
    return url + (f"#{anchor}" if anchor else "")


def rewrite(text, ref):
    # markdown images ![alt](x) and links [text](x)
    text = re.sub(r"(!?)\[([^\]]*)\]\(([^)\s]+)\)",
                  lambda m: f"{m[1]}[{m[2]}]({absolute(m[3], ref, bool(m[1]))})", text)
    # html <img src="x"> and <a href="x">
    text = re.sub(r'(<img\b[^>]*\bsrc=")([^"]+)"',
                  lambda m: f'{m[1]}{absolute(m[2], ref, True)}"', text)
    text = re.sub(r'(<a\b[^>]*\bhref=")([^"]+)"',
                  lambda m: f'{m[1]}{absolute(m[2], ref, False)}"', text)
    return text


if __name__ == "__main__":
    readme, ref = sys.argv[1], sys.argv[2]
    with open(readme, encoding="utf-8") as f:
        text = f.read()
    with open(readme, "w", encoding="utf-8") as f:
        f.write(rewrite(text, ref))
