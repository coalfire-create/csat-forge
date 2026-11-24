"""인쇄용 문제지 HTML 생성 (KaTeX 사용, 정답·해설은 별지)."""
from __future__ import annotations
import html

_CSS = """
body{font:15px/1.7 -apple-system,'Apple SD Gothic Neo','Noto Sans KR',sans-serif;max-width:780px;margin:24px auto;padding:0 16px;color:#111}
h1{font-size:22px;border-bottom:2px solid #111;padding-bottom:8px}
.q{margin:22px 0;page-break-inside:avoid}.no{font-weight:700;margin-right:6px}
.meta{color:#777;font-size:12px}.body{white-space:pre-wrap}
.ch{display:grid;grid-template-columns:repeat(5,auto);gap:4px 14px;margin-top:8px}
.ans{page-break-before:always}.ans li{margin:14px 0}.ex{white-space:pre-wrap;font-size:13px;color:#333}
@media print{body{margin:0}}
"""


def build_worksheet(records, title="수학 문제지") -> str:
    qs, ans = [], []
    for i, r in enumerate(records, 1):
        c, s = r["completion"], r["spec"]
        ch = ""
        if c.get("choices"):
            ch = '<div class="ch">' + "".join(
                f"<span>{'①②③④⑤'[k]} {html.escape(x)}</span>"
                for k, x in enumerate(c["choices"])) + "</div>"
        qs.append(f'<div class="q"><span class="no">{i}.</span>'
                  f'<span class="meta">[{html.escape(str(s.get("score", "")))}점 · '
                  f'{html.escape(str(s.get("topic", "")))}]</span>'
                  f'<div class="body">{html.escape(c["question"])}</div>{ch}</div>')
        ans.append(f'<li><b>{i}.</b> {html.escape(str(c["answer"]))}'
                   f'<div class="ex">{html.escape(c.get("explanation") or "")}</div></li>')
    return f"""<!doctype html><html lang="ko"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>{html.escape(title)}</title>
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.css">
<style>{_CSS}</style></head><body>
<h1>{html.escape(title)}</h1>{''.join(qs)}
<div class="ans"><h1>정답 및 해설</h1><ol style="list-style:none;padding:0">{''.join(ans)}</ol></div>
<script src="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/contrib/auto-render.min.js"></script>
<script>renderMathInElement(document.body,{{delimiters:[{{left:"$$",right:"$$",display:true}},{{left:"$",right:"$",display:false}}],throwOnError:false}})</script>
</body></html>"""
