"""Generate assets/vecta_demo.gif showing vecta-server startup, API calls, and Swagger UI."""

import os
from PIL import Image, ImageDraw, ImageFont

OUTPUT_PATH = "assets/vecta_demo.gif"
WIDTH = 840
HEIGHT = 500

# Color palette (modern dark mode)
BG_COLOR = (13, 17, 23)        # #0d1117
TERM_BG = (22, 27, 34)         # #161b22
HEADER_BG = (33, 38, 45)       # #21262d
BORDER_COLOR = (48, 54, 61)    # #30363d

COLOR_RED = (255, 95, 86)
COLOR_YELLOW = (255, 189, 46)
COLOR_GREEN = (39, 201, 63)

TEXT_WHITE = (230, 237, 243)
TEXT_MUTED = (139, 148, 158)
TEXT_CYAN = (88, 166, 255)
TEXT_GREEN = (63, 185, 80)
TEXT_YELLOW = (227, 179, 65)
TEXT_ORANGE = (240, 136, 62)
TEXT_PURPLE = (210, 168, 255)

def get_font(size=14, bold=False):
    # Try common monospace fonts on Windows/Linux
    candidates = [
        "C:/Windows/Fonts/consola.ttf",
        "C:/Windows/Fonts/consolab.ttf" if bold else "C:/Windows/Fonts/consola.ttf",
        "C:/Windows/Fonts/lucon.ttf",
        "DejaVuSansMono.ttf",
    ]
    for c in candidates:
        if os.path.exists(c):
            try:
                return ImageFont.truetype(c, size)
            except Exception:
                pass
    return ImageFont.load_default()

FONT = get_font(14)
FONT_BOLD = get_font(14, bold=True)
FONT_SM = get_font(12)
FONT_TITLE = get_font(16, bold=True)

def draw_window_frame(draw, title="vecta-server - bash (80x24)", is_browser=False):
    # Base canvas
    # Outer border & shadow
    draw.rectangle([(20, 20), (WIDTH - 20, HEIGHT - 20)], fill=TERM_BG, outline=BORDER_COLOR, width=1)
    
    # Title bar
    draw.rectangle([(20, 20), (WIDTH - 20, 56)], fill=HEADER_BG)
    draw.line([(20, 56), (WIDTH - 20, 56)], fill=BORDER_COLOR, width=1)
    
    # Traffic light dots
    draw.ellipse([(35, 34), (47, 46)], fill=COLOR_RED)
    draw.ellipse([(55, 34), (67, 46)], fill=COLOR_YELLOW)
    draw.ellipse([(75, 34), (87, 46)], fill=COLOR_GREEN)
    
    if is_browser:
        # Browser address bar
        draw.rectangle([(120, 28), (WIDTH - 40, 48)], fill=(13, 17, 23), outline=BORDER_COLOR, width=1)
        draw.text((130, 31), title, font=FONT_SM, fill=TEXT_CYAN)
    else:
        # Terminal title
        draw.text((105, 30), title, font=FONT_SM, fill=TEXT_MUTED)

def create_terminal_frame(lines):
    img = Image.new("RGB", (WIDTH, HEIGHT), BG_COLOR)
    draw = ImageDraw.Draw(img)
    draw_window_frame(draw)
    
    y = 75
    for line in lines:
        x = 40
        if isinstance(line, list):
            for part_text, part_color in line:
                draw.text((x, y), part_text, font=FONT, fill=part_color)
                bbox = FONT.getbbox(part_text)
                x += (bbox[2] - bbox[0])
        else:
            draw.text((x, y), line, font=FONT, fill=TEXT_WHITE)
        y += 24
    return img

def create_browser_frame():
    img = Image.new("RGB", (WIDTH, HEIGHT), BG_COLOR)
    draw = ImageDraw.Draw(img)
    draw_window_frame(draw, title="http://localhost:6333/docs - Vecta OpenAPI UI", is_browser=True)
    
    # Swagger Header banner
    draw.rectangle([(40, 75), (WIDTH - 40, 135)], fill=(22, 27, 34), outline=BORDER_COLOR)
    draw.text((55, 85), "⚡ Vecta Vector Database API", font=FONT_TITLE, fill=TEXT_WHITE)
    draw.rectangle([(320, 85), (370, 105)], fill=(31, 111, 235))
    draw.text((326, 88), "v0.1.0", font=FONT_SM, fill=TEXT_WHITE)
    draw.rectangle([(380, 85), (430, 105)], fill=(46, 160, 67))
    draw.text((388, 88), "OAS 3.0", font=FONT_SM, fill=TEXT_WHITE)
    draw.text((55, 112), "[ /api-docs/openapi.json ] - High-performance standalone vector engine", font=FONT_SM, fill=TEXT_MUTED)
    
    # Endpoint cards
    endpoints = [
        ("GET", "/health", "System health and runtime status probe", (46, 160, 67)),
        ("POST", "/collections", "Create a new vector collection (flat, ivf, hnsw, ivfpq)", (31, 111, 235)),
        ("GET", "/collections", "List all active collections and metadata", (46, 160, 67)),
        ("POST", "/collections/{name}/points", "Ingest vectors with IDs and metadata", (31, 111, 235)),
        ("POST", "/collections/{name}/search", "Query top-k approximate/exact nearest neighbors", (31, 111, 235)),
        ("POST", "/collections/{name}/checkpoint", "Force immediate WAL flush and snapshot to disk", (240, 136, 62)),
        ("DELETE", "/collections/{name}", "Drop collection and purge disk snapshot", (218, 54, 51)),
    ]
    
    y = 150
    for method, path, desc, color in endpoints:
        draw.rectangle([(40, y), (WIDTH - 40, y + 36)], fill=(13, 17, 23), outline=BORDER_COLOR)
        draw.rectangle([(40, y), (115, y + 36)], fill=color)
        draw.text((48, y + 9), f"{method:^6}", font=FONT_BOLD, fill=TEXT_WHITE)
        draw.text((130, y + 9), path, font=FONT_BOLD, fill=TEXT_WHITE)
        draw.text((380, y + 10), desc, font=FONT_SM, fill=TEXT_MUTED)
        y += 44
        
    return img

def build_demo_gif():
    frames = []
    durations = []
    
    # Scene 1: Startup
    s1_1 = [
        [("$ ", TEXT_MUTED), ("docker run -d -p 6333:6333 -v ./data:/data vecta", TEXT_WHITE)],
    ]
    s1_2 = s1_1 + [
        [("9f48a1c890123e42 (container started)", TEXT_MUTED)],
        [("$ ", TEXT_MUTED), ("cargo run --release --bin vecta-server", TEXT_WHITE)],
        [("    Finished", TEXT_GREEN), (" release [optimized] target(s) in 2.14s", TEXT_WHITE)],
        [("     Running", TEXT_GREEN), (" `target/release/vecta-server`", TEXT_WHITE)],
    ]
    s1_3 = s1_2 + [
        [("⚡ Vecta server listening on http://0.0.0.0:6333", TEXT_CYAN)],
        [("📁 Storage directory: ./data (WAL crash-recovery active)", TEXT_ORANGE)],
        [("📖 Swagger UI docs:  http://localhost:6333/docs", TEXT_PURPLE)],
        [("🔒 Authentication:   Bearer Token enabled", TEXT_GREEN)],
    ]
    
    frames.append(create_terminal_frame(s1_1)); durations.append(800)
    frames.append(create_terminal_frame(s1_2)); durations.append(1000)
    frames.append(create_terminal_frame(s1_3)); durations.append(1800)
    
    # Scene 2: Create Collection
    s2_1 = s1_3 + [
        [("", TEXT_WHITE)],
        [("$ ", TEXT_MUTED), ("curl -X POST http://localhost:6333/collections \\", TEXT_CYAN)],
        [("    -H ", TEXT_MUTED), ("'Authorization: Bearer test_secret' \\", TEXT_YELLOW)],
        [("    -d ", TEXT_MUTED), ("'{\"name\":\"articles\",\"dim\":4,\"index_type\":\"hnsw\"}'", TEXT_GREEN)],
    ]
    s2_2 = s2_1 + [
        [("HTTP/1.1 201 Created", TEXT_GREEN)],
        [("{\"name\":\"articles\",\"dim\":4,\"index_type\":\"hnsw\",\"metric\":\"euclidean\",\"vector_count\":0}", TEXT_WHITE)],
    ]
    frames.append(create_terminal_frame(s2_1)); durations.append(1200)
    frames.append(create_terminal_frame(s2_2)); durations.append(1800)
    
    # Scene 3: Insert Points
    s3_1 = [
        [("$ ", TEXT_MUTED), ("# Ingest 3 embedded document vectors", TEXT_MUTED)],
        [("$ ", TEXT_MUTED), ("curl -X POST http://localhost:6333/collections/articles/points \\", TEXT_CYAN)],
        [("    -H ", TEXT_MUTED), ("'Authorization: Bearer test_secret' \\", TEXT_YELLOW)],
        [("    -d ", TEXT_MUTED), ("'{\"id\": 1, \"vector\": [0.25, 0.50, 0.75, 1.0]}'", TEXT_GREEN)],
        [("HTTP/1.1 201 Created -> Point 1 indexed & WAL logged", TEXT_GREEN)],
        [("$ ", TEXT_MUTED), ("curl -X POST http://localhost:6333/collections/articles/points \\", TEXT_CYAN)],
        [("    -d ", TEXT_MUTED), ("'{\"id\": 2, \"vector\": [0.26, 0.49, 0.74, 0.99]}'", TEXT_GREEN)],
        [("HTTP/1.1 201 Created -> Point 2 indexed & WAL logged", TEXT_GREEN)],
    ]
    frames.append(create_terminal_frame(s3_1)); durations.append(2200)
    
    # Scene 4: Top-K Vector Search Query
    s4_1 = s3_1 + [
        [("", TEXT_WHITE)],
        [("$ ", TEXT_MUTED), ("# Query nearest neighbors with HNSW ef_search=64", TEXT_MUTED)],
        [("$ ", TEXT_MUTED), ("curl -X POST http://localhost:6333/collections/articles/search \\", TEXT_CYAN)],
        [("    -H ", TEXT_MUTED), ("'Authorization: Bearer test_secret' \\", TEXT_YELLOW)],
        [("    -d ", TEXT_MUTED), ("'{\"vector\": [0.25, 0.50, 0.75, 1.0], \"k\": 2, \"ef_search\": 64}'", TEXT_GREEN)],
    ]
    s4_2 = s4_1 + [
        [("HTTP/1.1 200 OK  [latency: 0.04ms]", TEXT_GREEN)],
        [("{\"results\":[{\"id\":1,\"score\":0.0},{\"id\":2,\"score\":0.0003}]}", TEXT_WHITE)],
    ]
    frames.append(create_terminal_frame(s4_1)); durations.append(1400)
    frames.append(create_terminal_frame(s4_2)); durations.append(2500)
    
    # Scene 5: Browser Swagger UI
    browser_frame = create_browser_frame()
    frames.append(browser_frame); durations.append(3500)
    
    # Save animated GIF
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    frames[0].save(
        OUTPUT_PATH,
        save_all=True,
        append_images=frames[1:],
        duration=durations,
        loop=0,
        optimize=True
    )
    print(f"Generated {OUTPUT_PATH} ({len(frames)} frames)")

if __name__ == "__main__":
    build_demo_gif()
