import os
from PIL import Image, ImageDraw, ImageFilter
import random
import numpy as np

def create_impressionist_image(width=128, height=128):
    """Impressionist style"""
    img = Image.new('RGB', (width, height))
    draw = ImageDraw.Draw(img)
    
    # s9ft color palette - pastels and light tones
    colors = [(173, 216, 230), (255, 182, 193), (221, 160, 221), 
              (152, 251, 152), (255, 218, 185), (176, 224, 230),
              (255, 228, 196), (230, 230, 250), (255, 240, 245)]
    
    # soft brushstroke-like texture
    for _ in range(50):
        color = random.choice(colors)
        x, y = random.randint(0, width), random.randint(0, height)
        size = random.randint(15, 40)
        draw.ellipse([x-size, y-size, x+size, y+size], fill=color)
    
    # more soft circular patches - impressionist style
    for _ in range(30):
        color = random.choice(colors)
        x, y = random.randint(0, width), random.randint(0, height)
        size = random.randint(8, 20)
        draw.ellipse([x-size, y-size, x+size, y+size], fill=color)
    
    # heavy blur - soft edges
    img = img.filter(ImageFilter.GaussianBlur(radius=3))
    return img

def create_cubist_image(width=128, height=128):
    """Cubist style"""
    img = Image.new('RGB', (width, height), color=(240, 240, 240))
    draw = ImageDraw.Draw(img)
    
    # bold, contrasting colors typical - cubism
    colors = [(255, 0, 0), (0, 0, 255), (255, 255, 0), 
              (0, 255, 0), (128, 0, 128), (255, 165, 0),
              (0, 128, 128), (128, 128, 0), (255, 0, 255)]
    
    # overlapping geometric shapes
    for _ in range(15):
        color = random.choice(colors)
        shape_type = random.choice(['rectangle', 'polygon', 'triangle'])
        
        if shape_type == 'rectangle':
            x1, y1 = random.randint(0, width), random.randint(0, height)
            x2, y2 = random.randint(0, width), random.randint(0, height)
            x1, x2 = min(x1, x2), max(x1, x2)
            y1, y2 = min(y1, y2), max(y1, y2)
            draw.rectangle([x1, y1, x2, y2], fill=color, outline='black', width=3)
        elif shape_type == 'triangle':
            points = [(random.randint(0, width), random.randint(0, height)) for _ in range(3)]
            draw.polygon(points, fill=color, outline='black', width=2)
        else:
            points = [(random.randint(0, width), random.randint(0, height)) for _ in range(random.randint(4, 6))]
            draw.polygon(points, fill=color, outline='black', width=2)
    
    return img

def create_surrealist_image(width=128, height=128):
    """Surrealist style"""
    img = Image.new('RGB', (width, height))
    draw = ImageDraw.Draw(img)
    
    # dream-like color combinations
    gradient_start = (random.randint(50, 200), random.randint(50, 200), random.randint(100, 255))
    gradient_end = (random.randint(100, 255), random.randint(50, 150), random.randint(50, 200))
    
    # gradient background
    for y in range(height):
        ratio = y / height
        r = int(gradient_start[0] * (1 - ratio) + gradient_end[0] * ratio)
        g = int(gradient_start[1] * (1 - ratio) + gradient_end[1] * ratio)
        b = int(gradient_start[2] * (1 - ratio) + gradient_end[2] * ratio)
        draw.line([(0, y), (width, y)], fill=(r, g, b))
    
    # organic shapes
    for _ in range(8):
        x, y = random.randint(0, width), random.randint(0, height)
        radius = random.randint(15, 35)
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        draw.ellipse([x-radius, y-radius, x+radius, y+radius], fill=color, outline=(0, 0, 0), width=2)
    
    # wavy lines - dream-like quality
    for _ in range(5):
        points = [(random.randint(0, width), random.randint(0, height)) for _ in range(4)]
        draw.line(points, fill=(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)), width=3)
    
    # slight blur - dream-like quality
    img = img.filter(ImageFilter.GaussianBlur(radius=1.5))
    return img

def create_renaissance_image(width=128, height=128):
    """Renaissance style"""
    img = Image.new('RGB', (width, height))
    draw = ImageDraw.Draw(img)
    
    # warm/earthy tones
    colors = [(139, 69, 19), (160, 82, 45), (205, 133, 63), 
              (222, 184, 135), (245, 222, 179), (210, 180, 140),
              (188, 143, 143), (165, 42, 42), (101, 67, 33)]
    
    # textured background
    for i in range(12):
        color = random.choice(colors)
        y_pos = int(height * i / 12)
        draw.rectangle([0, y_pos, width, y_pos + height // 12 + 5], fill=color)
    
    # classical central composition - concentric circles
    center_x, center_y = width // 2, height // 2
    for radius in range(35, 8, -4):
        color = random.choice(colors)
        draw.ellipse([center_x - radius, center_y - radius, 
                     center_x + radius, center_y + radius], 
                     fill=color, outline=(80, 50, 30), width=1)
    
    # vertical elements
    for _ in range(3):
        x = random.randint(10, width - 10)
        color = random.choice(colors)
        draw.rectangle([x, 0, x + 8, height], fill=color)
    
    return img

def create_abstract_image(width=128, height=128):
    """Abstract style"""
    # random bright background
    bg_colors = [(255, 255, 255), (0, 0, 0), (255, 200, 200), (200, 200, 255)]
    img = Image.new('RGB', (width, height), color=random.choice(bg_colors))
    draw = ImageDraw.Draw(img)
    
    # very vibrant/saturated colors
    colors = [(255, 0, 0), (0, 0, 255), (255, 255, 0), (0, 255, 0), 
              (255, 0, 255), (0, 255, 255), (128, 0, 128), (255, 165, 0),
              (255, 20, 147), (0, 191, 255), (50, 205, 50)]
    
    # abstract shapes
    for _ in range(20):
        color = random.choice(colors)
        shape = random.choice(['circle', 'line', 'arc', 'rectangle', 'polygon'])
        
        if shape == 'circle':
            x, y = random.randint(0, width), random.randint(0, height)
            r = random.randint(8, 35)
            draw.ellipse([x-r, y-r, x+r, y+r], fill=color)
        elif shape == 'line':
            x1, y1 = random.randint(0, width), random.randint(0, height)
            x2, y2 = random.randint(0, width), random.randint(0, height)
            draw.line([(x1, y1), (x2, y2)], fill=color, width=random.randint(4, 10))
        elif shape == 'rectangle':
            x1, y1 = random.randint(0, width), random.randint(0, height)
            x2, y2 = random.randint(0, width), random.randint(0, height)
            x1, x2 = min(x1, x2), max(x1, x2)
            y1, y2 = min(y1, y2), max(y1, y2)
            draw.rectangle([x1, y1, x2, y2], fill=color)
        elif shape == 'polygon':
            points = [(random.randint(0, width), random.randint(0, height)) for _ in range(random.randint(3, 6))]
            draw.polygon(points, fill=color)
        else:
            x1, y1 = random.randint(0, width), random.randint(0, height)
            x2, y2 = random.randint(0, width), random.randint(0, height)
            x1, x2 = min(x1, x2), max(x1, x2)
            y1, y2 = min(y1, y2), max(y1, y2)
            draw.arc([x1, y1, x2, y2], 0, random.randint(90, 360), fill=color, width=random.randint(3, 8))
    
    return img

def create_baroque_image(width=128, height=128):
    """Baroque style - dramatic, rich, ornate"""
    img = Image.new('RGB', (width, height))
    draw = ImageDraw.Draw(img)
    
    # rich, deep colors - gold, burgundy, deep blues
    colors = [(139, 0, 0), (75, 0, 130), (218, 165, 32), 
              (139, 69, 19), (128, 0, 0), (184, 134, 11),
              (85, 107, 47), (72, 61, 139), (148, 0, 211)]
    
    # dark dramatic background
    bg_color = random.choice([(20, 20, 40), (40, 20, 20), (20, 40, 20)])
    draw.rectangle([0, 0, width, height], fill=bg_color)
    
    # ornate circular patterns from center
    center_x, center_y = width // 2, height // 2
    for i in range(8):
        angle_offset = random.randint(0, 360)
        for j in range(6):
            angle = (j * 60 + angle_offset) * 3.14159 / 180
            radius = 20 + i * 5
            x = int(center_x + radius * np.cos(angle))
            y = int(center_y + radius * np.sin(angle))
            size = random.randint(3, 8)
            color = random.choice(colors)
            draw.ellipse([x-size, y-size, x+size, y+size], fill=color, outline=(218, 165, 32))
    
    # dramatic curved lines
    for _ in range(6):
        points = []
        for i in range(20):
            x = int(width * i / 20)
            y = int(height / 2 + 30 * np.sin(x * 0.1 + random.random() * 3))
            points.append((x, y))
        draw.line(points, fill=random.choice(colors), width=2)
    
    return img

def create_expressionist_image(width=128, height=128):
    """Expressionist style - bold, emotional, distorted"""
    img = Image.new('RGB', (width, height))
    draw = ImageDraw.Draw(img)
    
    # intense, emotional colors
    colors = [(255, 0, 0), (255, 140, 0), (255, 215, 0), 
              (0, 100, 0), (0, 0, 139), (128, 0, 128),
              (220, 20, 60), (255, 69, 0), (139, 0, 139)]
    
    # dramatic brushstroke-like background
    for i in range(15):
        color = random.choice(colors)
        x1, y1 = random.randint(0, width), random.randint(0, height)
        angle = random.uniform(0, 3.14159)
        length = random.randint(30, 80)
        x2 = int(x1 + length * np.cos(angle))
        y2 = int(y1 + length * np.sin(angle))
        draw.line([(x1, y1), (x2, y2)], fill=color, width=random.randint(8, 15))
    
    # distorted shapes
    for _ in range(10):
        color = random.choice(colors)
        points = [(random.randint(0, width), random.randint(0, height)) for _ in range(random.randint(3, 5))]
        draw.polygon(points, fill=color, outline=(0, 0, 0), width=3)
    
    # heavy texture
    for _ in range(100):
        x, y = random.randint(0, width), random.randint(0, height)
        color = random.choice(colors)
        draw.point((x, y), fill=color)
    
    return img

def create_pop_art_image(width=128, height=128):
    """Pop Art style - bright, bold, graphic, commercial"""
    img = Image.new('RGB', (width, height))
    draw = ImageDraw.Draw(img)
    
    # vivid pop art colors
    colors = [(255, 0, 0), (0, 0, 255), (255, 255, 0), 
              (255, 0, 255), (0, 255, 255), (255, 165, 0),
              (0, 255, 0), (255, 20, 147), (138, 43, 226)]
    
    # create grid pattern
    cell_size = width // 4
    for i in range(4):
        for j in range(4):
            x1, y1 = i * cell_size, j * cell_size
            x2, y2 = x1 + cell_size, y1 + cell_size
            color = random.choice(colors)
            draw.rectangle([x1, y1, x2, y2], fill=color, outline=(0, 0, 0), width=3)
            
            # add dots or circles in cells (Lichtenstein style)
            if random.random() > 0.5:
                for _ in range(5):
                    dot_x = random.randint(x1 + 5, x2 - 5)
                    dot_y = random.randint(y1 + 5, y2 - 5)
                    draw.ellipse([dot_x-3, dot_y-3, dot_x+3, dot_y+3], fill=(0, 0, 0))
    
    # bold outlines
    draw.rectangle([0, 0, width-1, height-1], outline=(0, 0, 0), width=4)
    
    return img

def create_minimalist_image(width=128, height=128):
    """Minimalist style - simple, clean, geometric"""
    img = Image.new('RGB', (width, height))
    draw = ImageDraw.Draw(img)
    
    # limited color palette - mostly neutral with one accent
    bg_colors = [(255, 255, 255), (245, 245, 245), (240, 240, 240)]
    accent_colors = [(0, 0, 0), (255, 0, 0), (0, 0, 255), (255, 200, 0)]
    
    bg = random.choice(bg_colors)
    accent = random.choice(accent_colors)
    
    draw.rectangle([0, 0, width, height], fill=bg)
    
    # very few simple shapes (1-3)
    num_shapes = random.randint(1, 3)
    for _ in range(num_shapes):
        shape_type = random.choice(['circle', 'rectangle', 'line'])
        
        if shape_type == 'circle':
            x, y = random.randint(width//4, 3*width//4), random.randint(height//4, 3*height//4)
            r = random.randint(15, 40)
            draw.ellipse([x-r, y-r, x+r, y+r], fill=accent)
        elif shape_type == 'rectangle':
            x1 = random.randint(width//4, width//2)
            y1 = random.randint(height//4, height//2)
            w = random.randint(20, 50)
            h = random.randint(20, 50)
            draw.rectangle([x1, y1, x1+w, y1+h], fill=accent)
        else:
            x1 = random.randint(10, width-10)
            is_horizontal = random.choice([True, False])
            if is_horizontal:
                draw.line([(10, x1), (width-10, x1)], fill=accent, width=random.randint(2, 5))
            else:
                draw.line([(x1, 10), (x1, height-10)], fill=accent, width=random.randint(2, 5))
    
    return img

def create_dataset(num_images_per_style=20):
    """dataset for all art styles"""
    styles = {
        'Abstract': create_abstract_image,
        'Baroque': create_baroque_image,
        'Cubist': create_cubist_image,
        'Expressionist': create_expressionist_image,
        'Impressionist': create_impressionist_image,
        'Minimalist': create_minimalist_image,
        'Pop Art': create_pop_art_image,
        'Renaissance': create_renaissance_image,
        'Surrealist': create_surrealist_image
    }
    
    total_images = 0
    for style_name, create_func in styles.items():
        style_dir = f'dataset/{style_name}'
        os.makedirs(style_dir, exist_ok=True)
        
        for i in range(num_images_per_style):
            img = create_func()
            img.save(f'{style_dir}/image_{i:03d}.png')
        
        total_images += num_images_per_style
        print(f'Created {num_images_per_style} images for {style_name}')
    
    print(f'\nTotal: {total_images} images across {len(styles)} styles')

if __name__ == '__main__':
    print("Creating expanded art dataset...")
    print("Generating 300 images per style for better accuracy...")
    create_dataset(num_images_per_style=300)
    print("Dataset creation complete!")
