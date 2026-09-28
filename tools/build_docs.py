"""Generate portable HTML from the repository's maintained Markdown documents."""
from html import escape
from pathlib import Path
import re
from urllib.parse import urlsplit

import markdown

ROOT = Path(__file__).resolve().parents[1]
SOURCES = [ROOT/'README.md', ROOT/'THIRD_PARTY.md', *sorted((ROOT/'docs').glob('*.md'))]
CSS = '''
:root{color-scheme:light;--ink:#192d38;--teal:#147e78;--muted:#59707d;--line:#dce6e8}
*{box-sizing:border-box}html{scroll-behavior:smooth;scroll-padding-top:70px}
body{margin:0;color:var(--ink);background:#f3f7f7;font:16px/1.8 system-ui,-apple-system,"Noto Sans CJK SC",sans-serif}
header{background:white;border-bottom:1px solid var(--line);padding:14px max(20px,calc((100vw - 1240px)/2));display:flex;gap:24px;align-items:center;flex-wrap:wrap}
header strong{letter-spacing:2px}a{color:var(--teal);text-underline-offset:3px}header a{text-decoration:none;font-size:14px}
.layout{max-width:1280px;margin:auto;display:grid;grid-template-columns:210px minmax(0,1fr);gap:32px;padding:32px 24px}
aside{align-self:start;position:sticky;top:20px;max-height:90vh;overflow:auto;font-size:13px}aside ul{list-style:none;padding-left:12px}aside>div>ul{padding-left:0}aside a{text-decoration:none}
main{min-width:0;background:white;border:1px solid var(--line);border-radius:16px;padding:32px 40px}
h1{font-size:32px;line-height:1.3;margin-top:0}h2{margin-top:38px;border-top:1px solid var(--line);padding-top:22px;font-size:23px}h3{font-size:19px}h1,h2,h3{letter-spacing:.2px}
p{margin:16px 0}blockquote{margin:20px 0;padding:10px 18px;border-left:4px solid var(--teal);background:#eff8f6;font-size:15px}blockquote p{margin:5px 0}
.table-wrap{overflow:auto;max-width:100%;margin:20px 0}table{border-collapse:collapse;font-size:14px;min-width:600px;width:100%}th,td{text-align:left;vertical-align:top;padding:11px 12px;border:1px solid var(--line)}th{background:#edf5f4}tr:nth-child(even){background:#fafcfc}
pre{overflow:auto;background:#152b36;color:#e3f1f2;padding:18px;border-radius:10px;font-size:13px;line-height:1.7}code{font-family:ui-monospace,SFMono-Regular,Consolas,monospace;overflow-wrap:anywhere}p code,td code,li code{font-size:.87em;background:#eef3f4;border-radius:3px;padding:2px 4px}pre code{background:none;overflow-wrap:normal}
img{max-width:100%}footer{color:var(--muted);font-size:12px;margin-top:40px}li{margin:6px 0}
@media(max-width:800px){.layout{display:block;padding:16px 10px}aside{position:static;max-height:220px;margin:0 10px 20px}main{padding:24px 18px}h1{font-size:26px}header{padding:12px 18px;gap:14px}}
@media print{body{background:white}.layout{display:block;padding:0}aside,header{display:none}main{border:0;padding:0}pre{white-space:pre-wrap;color:black;background:#f1f3f4}.table-wrap{overflow:visible}table{min-width:0}h2{break-after:avoid}tr{break-inside:avoid}}
'''


def build(source, target=None):
    target = target or source.with_suffix('.html')
    md = markdown.Markdown(extensions=['tables', 'fenced_code', 'toc'], extension_configs={'toc': {'toc_depth': '2-3'}})
    body = md.convert(source.read_text())
    def link(match):
        href = match.group(1)
        parts = urlsplit(href)
        if parts.scheme or parts.netloc or not parts.path or parts.path.startswith('/'):
            return match.group(0)
        dest = (source.parent/parts.path).resolve()
        if dest in SOURCES or dest.suffix == '.html':
            if dest in SOURCES:
                dest = dest.with_suffix('.html')
            import os
            href = os.path.relpath(dest, target.parent) + ('#'+parts.fragment if parts.fragment else '')
        else:
            # Source and JSON links also work when served by the console.
            href = 'https://github.com/kelo020304/piper-IK/blob/main/' + dest.relative_to(ROOT).as_posix()
            if parts.fragment:
                href += '#'+parts.fragment
        return 'href="'+escape(href, quote=True)+'"'
    body = re.sub(r'href="([^"]+)"', link, body)
    body = body.replace('<table>', '<div class="table-wrap"><table>').replace('</table>', '</table></div>')
    prefix = '../' if target.parent == ROOT/'docs' else ''
    nav = ' '.join(f'<a href="{prefix}{path}">{label}</a>' for path,label in [
        ('README.html','开始'),('docs/RL.html','RL'),('docs/PINOCCHIO.html','Pinocchio'),('docs/HARDWARE.html','真机'),('docs/VALIDATION.html','验证')])
    title = source.read_text().splitlines()[0].lstrip('# ')
    target.write_text(f'<!doctype html>\n<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{escape(title)}</title><style>{CSS}</style></head><body><header><strong>PIPER IK</strong>{nav}</header><div class="layout"><aside aria-label="章节导航">{md.toc}</aside><main>{body}<footer>由 {escape(source.relative_to(ROOT).as_posix())} 生成 · python tools/build_docs.py · 无需联网加载样式</footer></main></div></body></html>\n')
    print(target.relative_to(ROOT))


if __name__ == '__main__':
    for source in SOURCES:
        build(source)
    for name in ('HARDWARE', 'SIMULATORS'):
        build(ROOT/'docs'/f'{name}.md', ROOT/f'{name}.html')
