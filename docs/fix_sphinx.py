from argparse import ArgumentParser
from pathlib import Path

from bs4 import BeautifulSoup
from bs4.element import NavigableString, Tag


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("doxy", type=Path, help="directory with Sphinx HTML")
    doxy_path: Path = parser.parse_args().doxy

    for p in doxy_path.iterdir():
        if p.suffix != ".html":
            continue

        print(p)

        soup = BeautifulSoup(p.read_text(), "lxml")
        changed = False

        # Fix spaces around & and * tokens
        for w in soup.select("span.w"):
            sib = w.next_sibling
            if not isinstance(sib, Tag):
                continue

            match sib.name, sib.get("class"), sib.contents:
                case "span", ["p"], [Tag() as outer]:
                    match outer.contents:
                        case [NavigableString() as a] if a == "&" or set(a) == {"*"}:
                            placeholder = soup.new_tag("span")
                            w.replace_with(placeholder)
                            sib.replace_with(w)
                            placeholder.replace_with(sib)
                            changed = True
                        case _:
                            continue
                case _:
                    continue

        # Fix references like: <a class="reference internal">backend</a>::…
        for ref in soup.select("a.reference.internal"):
            if ref.string != "backend":
                continue

            nxt = ref.next_sibling
            if not isinstance(nxt, Tag) or nxt.string != "::":
                continue

            tag = soup.new_tag("span", class_="n")
            tag.append(soup.new_tag("span", class_="pre", string="backend"))
            ref.replace_with(tag)
            changed = True

        # Remove template parameter spaces in navigation links
        for code in soup.select(
            "li.toc-h2.nav-item.toc-entry "
            + "> a.reference.internal.nav-link "
            + "> code.docutils.literal.notranslate"
        ):
            for child in code.children:
                if not isinstance(child, NavigableString) or child != " ":
                    continue
                ante, post = child.previous_sibling, child.next_sibling
                assert isinstance(ante, Tag) and isinstance(post, Tag)
                ante, post = ante.string, post.string
                assert ante and post
                if ante[-1] in "<>" or post[-1] in "<>":
                    child.decompose()

        if changed:
            p.write_text(str(soup))

    for p in (doxy_path / "operations").iterdir():
        if p.suffix != ".html":
            continue

        print(p)

        soup = BeautifulSoup(p.read_text(), "lxml")
        changed = False

        for ref in soup.select("a.reference.internal"):
            content = ref.string
            if content not in {"Mask", "Vector"}:
                continue

            tag = soup.new_tag("span", class_="n")
            tag.append(soup.new_tag("span", class_="pre", string=content))
            ref.replace_with(tag)
            changed = True

        if changed:
            p.write_text(str(soup))


if __name__ == "__main__":
    main()
