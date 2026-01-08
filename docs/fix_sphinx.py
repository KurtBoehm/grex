from argparse import ArgumentParser
from pathlib import Path

from bs4 import BeautifulSoup
from bs4.element import NavigableString, Tag


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("doxy", type=Path)
    doxy_path: Path = parser.parse_args().doxy

    for p in doxy_path.iterdir():
        if p.suffix != ".html":
            continue

        print(p)

        soup = BeautifulSoup(p.read_text(), "lxml")
        changed = False

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

        if changed:
            p.write_text(str(soup))


if __name__ == "__main__":
    main()
