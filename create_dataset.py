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

def create_dataset(num_images_per_style=20):
    """dataset for all art styles"""
    styles = {
        'Impressionist': create_impressionist_image,
        'Cubist': create_cubist_image,
        'Surrealist': create_surrealist_image,
        'Renaissance': create_renaissance_image,
        'Abstract': create_abstract_image
    }
    
    for style_name, create_func in styles.items():
        style_dir = f'dataset/{style_name}'
        os.makedirs(style_dir, exist_ok=True)
        
        for i in range(num_images_per_style):
            img = create_func()
            img.save(f'{style_dir}/image_{i:03d}.png')
        
        print(f'Created {num_images_per_style} images for {style_name}')

if __name__ == '__main__':
    print("Creating art dataset...")
    print("Generating 100 images per style for better training...")
    create_dataset(num_images_per_style=100)
    print("Dataset creation complete! Total: 500 images")
