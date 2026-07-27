# Social preview card

`social-preview.png` (1280×640) is the image GitHub shows when the repo is
shared on X, Slack, Discord, LinkedIn, or anywhere else that reads OpenGraph
tags. Without it, every share renders as a grey box.

**Uploading it:** GitHub has no API for this. Repo → Settings → General →
Social preview → Edit → Upload an image.

**Editing it:** `social-preview.svg` is the source. Regenerate the PNG with:

```bash
cat > /tmp/card.html <<'HTML'
<!doctype html><html><head><meta charset="utf-8">
<style>html,body{margin:0;padding:0;background:#0D0D0F}</style></head>
<body><img src="social-preview.svg" width="1280" height="640"></body></html>
HTML
cp /tmp/card.html .github/card.html
"/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" --headless \
  --disable-gpu --hide-scrollbars --force-device-scale-factor=1 \
  --window-size=1280,640 --default-background-color=0D0D0F \
  --screenshot=.github/social-preview.png "file://$PWD/.github/card.html"
rm .github/card.html
```

Rendering through a browser rather than an image generator keeps the text
pixel-exact — a social card with a typo in it is worse than no card.

**Check any change at 500px wide** before uploading. That is the size it
actually appears at in a Slack or X unfurl, and it is where over-decorated
cards fall apart.

Palette: background `#0D0D0F`, headline `#F5F1EA`, accent `#C9884A` (a warm
leather tone — *valise* is a small travel case, and it stays clear of the
purple-on-black that every other AI tool uses).
