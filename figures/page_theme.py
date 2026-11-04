"""Force a page to the light palette on a white background (user, 2026-11-04: "make both of the
artifacts white background"): drop the dark-mode token blocks and set --bg to white."""
import re


def white(html):
    html = re.sub(r'@media \(prefers-color-scheme: dark\)\{ :root:not\(\[data-theme="light"\]\)\{.*?color-scheme:dark \} \}\n', '', html, flags=re.S)
    html = re.sub(r':root\[data-theme="dark"\]\{.*?color-scheme:dark \}\n', '', html, flags=re.S)
    html = html.replace('--bg:#f5f6f2;', '--bg:#ffffff;', 1)
    if 'color-scheme:light;--bg' not in html:
        html = html.replace(':root{\n  --bg:#ffffff;', ':root{\n  color-scheme:light;--bg:#ffffff;', 1)
    return html
