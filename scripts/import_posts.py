#!/usr/bin/env python3
"""
Перенос постов из публичной ленты Telegram-канала на сайт.

Что делает:
  1. Если бот сообщил о конкретном посте — ждёт, пока он появится в публичной ленте (до 5 минут).
  2. Читает https://t.me/s/<канал> (и страницы назад, до поста «since» из настроек).
  3. Берёт текстовые посты: текст с жирным/курсивом/ссылками, дату, фото.
  4. Склеивает части (по списку «merge» в настройках и по пометкам «ч.1», «ч.2» в заголовке).
  5. Считает время чтения, скачивает фото, увеличивает их (если есть модель) и сохраняет в WebP.
  6. Записывает data/posts.json (состояние) и js/posts.js (то, что читает сайт).

Настройки и ручные правки — в файле posts-settings.json в корне сайта.
"""
import json, math, os, re, sys, time, io
from datetime import datetime, timezone, timedelta
import requests
from bs4 import BeautifulSoup, NavigableString, Tag
from PIL import Image, ImageOps

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SETTINGS = json.load(open(os.path.join(ROOT, 'posts-settings.json'), encoding='utf-8'))
CHANNEL = SETTINGS.get('channel', 'directorofcircus')
SINCE = int(SETTINGS.get('since', 1))
MIN_WORDS = int(SETTINGS.get('minWords', 80))
TZ = timezone(timedelta(hours=int(SETTINGS.get('utcOffsetHours', 3))))
STATE_PATH = os.path.join(ROOT, 'data', 'posts.json')
JS_PATH = os.path.join(ROOT, 'js', 'posts.js')
PHOTO_DIR = os.path.join(ROOT, 'assets', 'photos')
MODEL = os.path.join(ROOT, 'scripts', 'realesr-general-x4v3.onnx')
UA = {'User-Agent': 'Mozilla/5.0 (compatible; circus-director-site/1.0)'}
PART_RE = re.compile(r'[\s,.:–—-]*\(?\s*(ч(асть)?|п(ункт)?)\.?\s*\d+\s*\)?\s*$', re.I)
TARGET = int(os.environ.get('MESSAGE_ID') or 0)   # номер поста, о котором сообщил бот

def log(*a): print(*a, flush=True)

# ---------- чтение ленты ----------
def fetch_page(before=None):
    url = f'https://t.me/s/{CHANNEL}' + (f'?before={before}' if before else '')
    for attempt in range(4):
        try:
            r = requests.get(url, headers=UA, timeout=30)
            if r.status_code == 200: return r.text
        except requests.RequestException as e:
            log('  сеть:', e)
        time.sleep(3 * (attempt + 1))
    raise SystemExit('Не удалось открыть ленту ' + url)

def clean_inline(node):
    """HTML текста поста → наш безопасный HTML (strong/em/u/s/a), строки разделены '\\n'."""
    out = []
    for ch in node.children:
        if isinstance(ch, NavigableString):
            out.append(str(ch).replace('<', '&lt;').replace('>', '&gt;')); continue
        if not isinstance(ch, Tag): continue
        name = ch.name
        cls = ch.get('class') or []
        if name == 'br': out.append('\n'); continue
        if name == 'i' and 'emoji' in cls: out.append(ch.get_text()); continue
        inner = clean_inline(ch)
        if name in ('b', 'strong'): out.append(f'<strong>{inner}</strong>' if inner.strip() else inner)
        elif name in ('i', 'em'): out.append(f'<em>{inner}</em>' if inner.strip() else inner)
        elif name == 'u': out.append(f'<u>{inner}</u>')
        elif name in ('s', 'del'): out.append(f'<s>{inner}</s>')
        elif name == 'a':
            href = ch.get('href', '')
            if href.startswith('http'):
                out.append(f'<a href="{href}" target="_blank" rel="noopener">{inner}</a>')
            else: out.append(inner)
        elif name in ('h1', 'h2', 'h3', 'h4'): out.append('\n<strong>' + re.sub(r'</?strong>', '', inner).strip() + '</strong>\n')
        elif name == 'blockquote': out.append('\n' + inner + '\n')
        else: out.append(inner)
    return ''.join(out)

def balance(line):
    """Закрыть/открыть теги, разорванные переносом строки."""
    for tag in ('strong', 'em', 'u', 's'):
        o, c = line.count(f'<{tag}>'), line.count(f'</{tag}>')
        if o > c: line = line + f'</{tag}>' * (o - c)
        elif c > o: line = f'<{tag}>' * (c - o) + line
    return line

def parse_page(html):
    soup = BeautifulSoup(html, 'html.parser')
    msgs = []
    for w in soup.select('div.tgme_widget_message[data-post]'):
        try: mid = int(w['data-post'].split('/')[-1])
        except ValueError: continue
        t = w.select_one('a.tgme_widget_message_date time') or w.select_one('time[datetime]')
        dt = datetime.fromisoformat(t['datetime']).astimezone(TZ) if t and t.get('datetime') else None
        text_el = w.select_one('div.tgme_widget_message_text') or w.select_one('.js-message_text')
        html_text = clean_inline(text_el) if text_el else ''
        if not re.sub('<[^>]+>', '', html_text).strip():
            # запасной путь: текст в непривычной разметке — берём весь блок поста без служебных частей
            bubble = w.select_one('.tgme_widget_message_bubble')
            if bubble:
                b = BeautifulSoup(str(bubble), 'html.parser')
                for junk in b.select('.tgme_widget_message_author, .tgme_widget_message_footer, .tgme_widget_message_info, '
                                     '.tgme_widget_message_reactions, .tgme_widget_message_owner_name, .tgme_widget_message_views, '
                                     '.tgme_widget_message_forwarded_from, .tgme_widget_message_reply, '
                                     'a.tgme_widget_message_photo_wrap, .tgme_widget_message_video_player, '
                                     '.tgme_widget_message_link_preview, .message_media_not_supported_wrap, script, style'):
                    junk.decompose()
                inner = b.select_one('.tgme_widget_message_bubble')
                for blk in inner.find_all(['p', 'div', 'h1', 'h2', 'h3', 'h4', 'li', 'blockquote']):
                    blk.insert_after(NavigableString('\n'))
                html_text = clean_inline(inner)
            if TARGET and mid == TARGET:
                log(f'  Разметка поста {mid} (начало): ' + re.sub(r'\s+', ' ', str(w))[:1500])
        photos = []
        for a in w.select('a.tgme_widget_message_photo_wrap'):
            m = re.search(r"background-image:url\('([^']+)'\)", a.get('style', ''))
            if m: photos.append(m.group(1))
        msgs.append({'id': mid, 'date': dt, 'html': html_text, 'photos': photos})
    return msgs

def wait_for_target():
    """Бот сообщил о посте — ждём, пока он появится в публичной ленте (до 5 минут)."""
    if not TARGET: return
    for attempt in range(16):
        msgs = parse_page(fetch_page())
        top = max((m['id'] for m in msgs), default=0)
        if top >= TARGET:
            log(f'Пост {TARGET} есть в ленте.'); return
        log(f'Поста {TARGET} в ленте пока нет (последний {top}), жду 20 с…')
        time.sleep(20)
    log(f'Пост {TARGET} так и не появился в ленте за 5 минут — сработает ежедневная проверка.')

def read_channel():
    wait_for_target()
    seen, before, pages = {}, None, 0
    while True:
        msgs = parse_page(fetch_page(before)); pages += 1
        if not msgs: break
        for m in msgs: seen[m['id']] = m
        lowest = min(m['id'] for m in msgs)
        if lowest <= SINCE or pages > 40: break
        before = lowest
        time.sleep(1)
    log(f'Лента: {len(seen)} сообщений, страниц: {pages}')
    return [seen[k] for k in sorted(seen) if k >= SINCE]

# ---------- сборка постов ----------
def lines_of(html):
    return [balance(x.strip()) for x in html.split('\n') if re.sub('<[^>]+>', '', x).strip()]

def plain(s): return re.sub('<[^>]+>', '', s).strip()

def is_title_line(line):
    p = plain(line)
    return 0 < len(p) <= 90 and re.fullmatch(r'\s*<strong>.*</strong>\s*', line or '') is not None

def base_title(line):
    return PART_RE.sub('', plain(line)).strip(' .:—-')

def word_count(lines): return sum(len(plain(x).split()) for x in lines)

def build_posts(msgs):
    S = SETTINGS
    merge_groups = [list(map(str, g)) for g in S.get('merge', [])]
    in_group = {mid: g for g in merge_groups for mid in g}
    exclude = set(map(str, S.get('exclude', []))); include = set(map(str, S.get('include', [])))
    titles = {str(k): v for k, v in S.get('titles', {}).items()}
    by_id = {str(m['id']): m for m in msgs}
    ids = [str(m['id']) for m in msgs]
    newest = max((int(x) for x in ids), default=0)
    used, posts = set(), []
    i = 0
    while i < len(ids):
        mid = ids[i]
        if mid in used: i += 1; continue
        group = [x for x in in_group.get(mid, [mid]) if x in by_id]
        # автосклейка: следующий пост с тем же заголовком и пометкой «ч.N»
        if mid not in in_group:
            first = lines_of(by_id[mid]['html'])[:1]
            if first and is_title_line(first[0]) and PART_RE.search(plain(first[0])):
                b = base_title(first[0]); j = i + 1
                while j < len(ids):
                    nxt = lines_of(by_id[ids[j]]['html'])[:1]
                    if nxt and is_title_line(nxt[0]) and base_title(nxt[0]) == b and PART_RE.search(plain(nxt[0])):
                        group.append(ids[j]); j += 1
                    elif not by_id[ids[j]]['html'].strip():   # фото-сообщение альбома без текста
                        j += 1
                    else: break
        for g in group: used.add(g)
        i += 1
        if mid in exclude: continue
        lines, photos, title_line = [], [], None
        for g in group:
            m = by_id[g]; ls = lines_of(m['html'])
            if ls and is_title_line(ls[0]):
                if title_line is None: title_line = ls[0]
                ls = ls[1:]
            lines += ls; photos += m['photos']
        # фото альбома приходят отдельными сообщениями без текста сразу после поста
        j = ids.index(group[-1]) + 1
        while j < len(ids) and not by_id[ids[j]]['html'].strip() and by_id[ids[j]]['photos'] and ids[j] not in in_group:
            photos += by_id[ids[j]]['photos']; used.add(ids[j]); j += 1
        words = word_count(lines)
        if words < MIN_WORDS and mid not in include:
            if int(mid) == TARGET or int(mid) > newest - 3:
                log(f'  Пост {mid}: {words} слов — меньше {MIN_WORDS}, не переношу. '
                    f'Чтобы перенести, добавь "{mid}" в include в posts-settings.json.')
            continue
        if not lines: continue
        title = titles.get(mid) or (base_title(title_line) if title_line else None)
        if not title:
            p0 = plain(lines[0]); m = re.search(r'^(.{12,80}?[.!?…])(\s|$)', p0)
            title = (m.group(1) if m else (p0[:70].rsplit(' ', 1)[0] + '…')).rstrip('.')
        paras = []
        for ln in lines:
            p = plain(ln)
            if re.fullmatch(r'\s*<strong>.*</strong>\s*', ln) and len(p) <= 60 and not p.endswith(('.', '!', '?', '…')):
                paras.append(f'<h3>{p}</h3>')          # короткая жирная строка — подзаголовок
            else:
                paras.append(ln)
        d = by_id[mid]['date']
        posts.append({'id': mid, 'parts': group, 'title': title, 'date': d.strftime('%Y-%m-%d') if d else '',
                      'min': max(1, math.ceil(words / 180)), 'paras': paras, 'photo_urls': photos})
    posts.sort(key=lambda p: int(p['id']), reverse=True)
    return posts

# ---------- фото ----------
_session = None
def upscale(img):
    global _session
    if not os.path.exists(MODEL): return None
    import numpy as np, onnxruntime as ort
    if _session is None: _session = ort.InferenceSession(MODEL, providers=['CPUExecutionProvider'])
    a = np.asarray(img.convert('RGB')).astype(np.float32) / 255.
    H, W, _ = a.shape; T, P, S = 256, 12, 4
    out = np.zeros((H * S, W * S, 3), np.float32)
    for y in range(0, H, T):
        for x in range(0, W, T):
            y0, x0 = max(0, y - P), max(0, x - P); y1, x1 = min(H, y + T + P), min(W, x + T + P)
            r = _session.run(None, {'input': np.ascontiguousarray(a[y0:y1, x0:x1].transpose(2, 0, 1)[None])})[0][0].transpose(1, 2, 0)
            oy, ox = (y - y0) * S, (x - x0) * S; h, w = min(T, H - y) * S, min(T, W - x) * S
            out[y * S:y * S + h, x * S:x * S + w] = r[oy:oy + h, ox:ox + w]
    return Image.fromarray((np.clip(out, 0, 1) * 255 + .5).astype('uint8'))

def layout(k):
    if k == 1: return [1]
    if k == 2: return [1, 1]
    rows = [1] if k % 2 else []
    return rows + [2] * ((k - (1 if k % 2 else 0)) // 2)

def process_photos(post):
    urls = post.pop('photo_urls', [])
    if not urls: return None
    rows = layout(len(urls)); widths = []
    for r in rows: widths += [1440 if r == 1 else 760] * r
    os.makedirs(PHOTO_DIR, exist_ok=True)
    items = []
    for k, url in enumerate(urls):
        rel = f"assets/photos/{post['id']}-{k + 1}.webp"; path = os.path.join(ROOT, rel)
        if os.path.exists(path):
            with Image.open(path) as im: items.append({'src': rel, 'w': im.width, 'h': im.height}); continue
        try:
            src = Image.open(io.BytesIO(requests.get(url, headers=UA, timeout=60).content))
            src = ImageOps.exif_transpose(src).convert('RGB')
        except Exception as e:
            log('  фото не скачалось:', url, e); continue
        w = widths[k]; h = round(src.height * w / src.width)
        up = upscale(src) if src.width < w else None
        if up is not None:
            a = up.resize((w, h), Image.LANCZOS); b = src.resize((w, h), Image.LANCZOS)
            im = Image.blend(b, a, 0.7)
        else:
            im = src.resize((w, h), Image.LANCZOS) if src.width > w else src
        im.save(path, 'WEBP', quality=78, method=6)
        items.append({'src': rel, 'w': im.width, 'h': im.height})
        log('  фото', rel)
    return {'rows': layout(len(items)), 'items': items} if items else None

# ---------- запись ----------
def main():
    old = {}
    if os.path.exists(STATE_PATH):
        for p in json.load(open(STATE_PATH, encoding='utf-8'))['posts']: old[p['id']] = p
    msgs = read_channel()
    if not msgs: raise SystemExit('В ленте не нашлось сообщений — ничего не меняю.')
    posts = build_posts(msgs)
    for p in posts:
        prev = old.get(p['id'])
        photos = process_photos(p)
        if photos: p['photos'] = photos
        elif prev and prev.get('photos'): p['photos'] = prev['photos']
    # посты старше «since» сохраняем как были
    ids = {p['id'] for p in posts}
    for pid, p in old.items():
        if pid not in ids and int(pid) < SINCE: posts.append(p)
    posts.sort(key=lambda p: int(p['id']), reverse=True)
    if TARGET and str(TARGET) not in ids and not any(str(TARGET) in p.get('parts', []) for p in posts):
        log(f'Пост {TARGET} на сайт не попал (см. причину выше).')
    os.makedirs(os.path.dirname(STATE_PATH), exist_ok=True)
    json.dump({'posts': posts}, open(STATE_PATH, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    data = [{k: p[k] for k in ('id', 'title', 'date', 'min', 'paras', 'photos') if k in p} for p in posts]
    open(JS_PATH, 'w', encoding='utf-8').write(
        '// Публикации. Файл обновляется автоматически программой переноса из Telegram.\n'
        'window.POSTS_DATA = ' + json.dumps(data, ensure_ascii=False) + ';\n')
    log(f'Готово: {len(posts)} публикаций.')
    for p in posts[:5]: log('  ', p['id'], p['date'], p['title'])

if __name__ == '__main__':
    main()
