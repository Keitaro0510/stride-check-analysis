"""ステップ6：app/index.html に data/app_data.js、model.js、i18n.js を埋め込み、1つの HTML にする。

  outputs/stride_check.html     最初に日本語で表示する版
  outputs/stride_check_en.html  最初に英語で表示する版（どちらも画面の EN / 日本語 で切り替えられる）

外部に読みに行くのは Google Fonts だけ。アーティファクトとして公開するときも、このファイルをそのまま使う。
"""
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
APP = HERE / "app"
OUTS = {"ja": HERE / "outputs" / "stride_check.html", "en": HERE / "outputs" / "stride_check_en.html"}


def main():
    html = (APP / "index.html").read_text(encoding="utf-8")
    # 動画の解析はアーティファクトでは動かせない（外部のモデルや WebAssembly を読めない）ので、入口ごと外す
    extract_tag = '<script src="extract/extract.js"></script>'
    assert html.count(extract_tag) == 1
    html = html.replace(extract_tag, "")

    def inline(match):
        src = match.group(1)
        code = (APP / src).read_text(encoding="utf-8").replace("</script", "<\\/script")
        return f"<script>/* {src} */\n{code}</script>"

    html, n = re.subn(r'<script src="([^"]+)"></script>', inline, html)
    assert n == 3, f"埋め込むスクリプトが3つでない（{n}）"
    external = sorted(set(re.findall(r'(?:src|href)="(https?://[^"]+)"', html)))
    assert all(u.startswith(("https://fonts.googleapis.com", "https://fonts.gstatic.com")) for u in external), external
    for lang, out in OUTS.items():
        page = html.replace('data-default-lang="ja"', f'data-default-lang="{lang}"').replace('<html lang="ja"', f'<html lang="{lang}"')
        out.parent.mkdir(exist_ok=True)
        out.write_text(page, encoding="utf-8")
        print(f"{out.relative_to(HERE)}: {out.stat().st_size / 1024:.0f} KB")
    print("外部の読み込み: " + ", ".join(external))

if __name__ == "__main__":
    main()
