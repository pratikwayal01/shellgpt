#!/usr/bin/env python3
"""
Fetch tldr-pages command examples and build tldr_corpus.txt.

Extracts every backtick-quoted command line from the English pages/
directory of https://github.com/tldr-pages/tldr, dedupes, writes one
command per line. train() picks it up via build_corpus().

Run:  python fetch_tldr.py
"""
import io, re, tarfile, urllib.request
from pathlib import Path

URL  = "https://github.com/tldr-pages/tldr/archive/refs/heads/main.tar.gz"
OUT  = Path("tldr_corpus.txt")
CMD  = re.compile(r"^`([^`]+)`")          # a quoted example command line
PAGE = re.compile(r"tldr-main/pages/[^/]+/[^/]+\.md$")   # English only

def main():
    print("Downloading tldr-pages...")
    data = urllib.request.urlopen(URL).read()
    commands, seen = [], set()
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tf:
        for member in tf.getmembers():
            if not member.isfile() or not PAGE.match(member.name):
                continue
            text = tf.extractfile(member).read().decode("utf-8", "replace")
            for line in text.splitlines():
                m = CMD.match(line.strip())
                if m:
                    cmd = m.group(1)
                    if cmd not in seen:
                        seen.add(cmd)
                        commands.append(cmd)
    OUT.write_text("\n".join(commands) + "\n")
    print(f"Wrote {len(commands)} unique commands -> {OUT}")

if __name__ == "__main__":
    main()