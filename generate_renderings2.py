import subprocess
from pathlib import Path
from PIL import Image
import numpy as np
import random
from IPython.display import display
from tqdm import tqdm
import requests
import shutil
from urllib.parse import urlparse, parse_qsl, urlencode, urlunparse

# ------------------- DOWNLOAD STL FILES -------------------
data_dir  = "https://www.dropbox.com/scl/fo/xnu0e0c0udo4qe5mn5gvu/AEvcg5S8PjhBBoBerNQ5ANY?rlkey=31i9mphl5w419kjccpiqzf3s2&st=5o7us7wz&dl=0"
source_root = Path("data/STL_Files")
target_root = Path("data/Renderings") # root output for images

def _to_direct_dropbox(url):
    parsed = urlparse(url)
    query = dict(parse_qsl(parsed.query))
    query["dl"] = "1"
    return urlunparse(parsed._replace(query=urlencode(query)))

download_url = _to_direct_dropbox(data_dir)
zip_path = source_root.parent / "source_dataset.zip"
source_root.mkdir(parents=True, exist_ok=True)
target_root.mkdir(parents=True, exist_ok=True)

print(f"Downloading STL archive from Dropbox to {zip_path}...")
with requests.get(download_url, stream=True) as resp:
    resp.raise_for_status()
    total_size = int(resp.headers.get('content-length', 0))
    
    with open(zip_path, "wb") as fh:
        with tqdm(total=total_size, unit='B', unit_scale=True, desc="Downloading") as pbar:
            for chunk in resp.iter_content(chunk_size=8192):
                if chunk:
                    fh.write(chunk)
                    pbar.update(len(chunk))

print("Extracting archive...")
shutil.unpack_archive(str(zip_path), str(source_root))
zip_path.unlink()
print('✓ Files downloaded and extracted.')

# ------------------- CONFIG -------------------
BLENDER_PATH = "/Applications/Blender.app/Contents/MacOS/Blender"

NUM_VIEWS = 30         # images per part
RESOLUTION = 512
SAMPLES = 64

# Post-render augmentation
GAUSSIAN_NOISE_STD = 10
CROP_RATIO = 0.9

# ------------------- BLENDER SCRIPT TEMPLATE -------------------
BLENDER_SCRIPT = """
import bpy
from mathutils import Vector
import math
import random
from pathlib import Path

stl_dir = Path(r"{stl_dir}")
output_dir = Path(r"{output_dir}")
num_views = {num_views}
resolution = {resolution}
samples = {samples}

stl_files = sorted(stl_dir.glob("*.stl"))
if not stl_files:
    print(f"No STL files found in {{stl_dir}}")
    quit()

print(f"Found {{len(stl_files)}} STL files in {{stl_dir}}")

def clear_scene():
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()

def join_meshes(meshes):
    # Join multiple meshes into a single object if needed
    if len(meshes) == 1:
        return meshes[0]
    bpy.ops.object.select_all(action='DESELECT')
    for m in meshes:
        m.select_set(True)
    bpy.context.view_layer.objects.active = meshes[0]
    bpy.ops.object.join()
    return bpy.context.view_layer.objects.active

def center_and_scale(obj, target_size=2.0):
    # Put origin at bounds center and move to origin
    bpy.ops.object.select_all(action='DESELECT')
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj

    bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='BOUNDS')
    obj.location = (0.0, 0.0, 0.0)

    # Compute max dimension in world space
    bbox = [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]
    max_dim = max(
        max(v.x for v in bbox) - min(v.x for v in bbox),
        max(v.y for v in bbox) - min(v.y for v in bbox),
        max(v.z for v in bbox) - min(v.z for v in bbox)
    )

    if max_dim > 0:
        scale = target_size / max_dim
        obj.scale = (scale, scale, scale)
        bpy.ops.object.transform_apply(scale=True, location=True, rotation=True)

def color_object_orange(obj):
    # Create new Principled material
    mat = bpy.data.materials.new(name="PartOrange")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    if bsdf is None:
        bsdf = mat.node_tree.nodes.new(type="ShaderNodeBsdfPrincipled")

    # Vivid orange
    bsdf.inputs["Base Color"].default_value = (1.0, 0.35, 0.0, 1.0)
    bsdf.inputs["Roughness"].default_value = 0.4
    bsdf.inputs["Metallic"].default_value = 0.0

    # Remove any existing materials and assign this one
    obj.data.materials.clear()
    obj.data.materials.append(mat)

def add_reference_cube(obj):
    # Fixed cube size = 0.25 meters = 25 cm
    cube_size = 0.25

    bpy.ops.mesh.primitive_cube_add(size=cube_size)
    cube = bpy.context.object

    # Compute bounding box AFTER scaling/centering
    bbox = [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]
    min_x = min(v.x for v in bbox)
    min_y = min(v.y for v in bbox)
    min_z = min(v.z for v in bbox)

    # Place cube near bottom-left-front corner (slightly offset)
    margin = cube_size * 0.5
    cube.location = (
        min_x - cube_size - margin,
        min_y - cube_size - margin,
        min_z + cube_size * 0.5,
    )

    # Parent cube to part so it rotates with it
    cube.parent = obj
    cube.matrix_parent_inverse = obj.matrix_world.inverted()

    return cube

def setup_camera_and_lights():
    scene = bpy.context.scene

    # Camera: fixed distance, always looking at origin
    camera_distance = 5.0
    bpy.ops.object.camera_add(location=(camera_distance, -camera_distance, camera_distance))
    camera = bpy.context.object
    direction = Vector((0, 0, 0)) - camera.location
    camera.rotation_euler = direction.to_track_quat('-Z', 'Y').to_euler()
    scene.camera = camera

    # Lighting
    def add_light(loc, energy):
        bpy.ops.object.light_add(type='AREA', location=loc)
        light = bpy.context.object
        light.data.energy = energy
        return light

    key_light = add_light((4, -4, 5), 300)
    fill_light = add_light((-3, -2, 3), 150)
    back_light = add_light((0, 4, 4), 200)

    return camera, key_light, fill_light, back_light

def set_random_background():
    scene = bpy.context.scene
    if scene.world is None:
        scene.world = bpy.data.worlds.new("World")
    world = scene.world
    world.use_nodes = True

    bg = world.node_tree.nodes.get("Background")
    if bg is None:
        bg = world.node_tree.nodes.new(type="ShaderNodeBackground")
        world_output = world.node_tree.nodes.get("World Output")
        if world_output is None:
            world_output = world.node_tree.nodes.new(type="ShaderNodeOutputWorld")
        world.node_tree.links.new(bg.outputs[0], world_output.inputs[0])

    # Only dark backgrounds for strong contrast with orange part
    palette = [
        (0.05, 0.05, 0.05, 1.0),   # almost black
        (0.15, 0.15, 0.15, 1.0),   # deep gray
        (0.30, 0.30, 0.30, 1.0),   # dark gray
    ]
    color = random.choice(palette)
    bg.inputs["Color"].default_value = color

# Render settings
scene = bpy.context.scene
scene.render.engine = 'CYCLES'
scene.cycles.samples = samples
scene.cycles.use_denoising = True
scene.render.resolution_x = resolution
scene.render.resolution_y = resolution
scene.render.image_settings.file_format = 'JPEG'

for stl_path in stl_files:
    print(f"Processing: {{stl_path.name}}")

    clear_scene()

    # Import STL
    bpy.ops.import_mesh.stl(filepath=str(stl_path))
    meshes = [obj for obj in bpy.context.selected_objects if obj.type == 'MESH']
    if not meshes:
        print(f"  ✗ No meshes found in {{stl_path.name}}")
        continue

    # Join meshes into a single object if necessary
    obj = join_meshes(meshes)

    # Normalize part size and position
    center_and_scale(obj)

    # Color the part orange
    color_object_orange(obj)

    # Add size-reference cube AFTER scaling & coloring
    cube = add_reference_cube(obj)

    # Camera/lights
    camera, key_light, fill_light, back_light = setup_camera_and_lights()

    part_name = stl_path.stem

    # Make a folder for this part
    part_output_dir = Path(output_dir) / part_name
    part_output_dir.mkdir(parents=True, exist_ok=True)

    for i in range(num_views):
        # ------------- RANDOM ORIENTATION -------------
        rx = math.radians(random.uniform(-45, 45))
        ry = math.radians(random.uniform(-45, 45))
        rz = math.radians(random.uniform(0, 360))
        obj.rotation_euler = (rx, ry, rz)

        # ------------- RANDOM LIGHTS (SHADOWS) -------------
        key_light.location = (
            4 + random.uniform(-2, 2),
            -4 + random.uniform(-2, 2),
            5 + random.uniform(-1, 1),
        )
        fill_light.location = (
            -3 + random.uniform(-1.5, 1.5),
            -2 + random.uniform(-1.5, 1.5),
            3 + random.uniform(-1, 1),
        )
        back_light.location = (
            0 + random.uniform(-2, 2),
            4 + random.uniform(-2, 2),
            4 + random.uniform(-1, 1),
        )

        key_light.data.energy = 300 + random.uniform(-120, 120)
        fill_light.data.energy = 150 + random.uniform(-60, 60)
        back_light.data.energy = 200 + random.uniform(-80, 80)

        # ------------- RANDOM DARK BACKGROUND -------------
        set_random_background()

        filepath = part_output_dir / f"{{part_name}}_view_{{i:02d}}.jpg"
        scene.render.filepath = str(filepath)
        bpy.ops.render.render(write_still=True)
        print(f"  Rendered view {{i+1}}/{{num_views}} → {{filepath}}")

print("✓ All STL files rendered!")
"""

# ------------------- SAVE AND RUN BLENDER -------------------
script_path = target_root / "temp_blender_render.py"
with open(script_path, "w") as f:
    f.write(BLENDER_SCRIPT.format(
        stl_dir=str(source_root),
        output_dir=str(target_root),
        num_views=NUM_VIEWS,
        resolution=RESOLUTION,
        samples=SAMPLES
    ))

# Run Blender headless
subprocess.run([BLENDER_PATH, "--background", "--python", str(script_path)])

# ------------------- POST-PROCESS AUGMENTATION -------------------
def augment_image(image_path: Path):
    img = Image.open(image_path)
    arr = np.array(img).astype(np.float32)

    noise_std = GAUSSIAN_NOISE_STD
    noise = np.random.normal(0, noise_std, arr.shape)
    arr_noisy = np.clip(arr + noise, 0, 255).astype(np.uint8)
    img_noisy = Image.fromarray(arr_noisy)

    # Random crop
    w, h = img_noisy.size
    crop_size = int(CROP_RATIO * w)
    left = random.randint(0, w - crop_size)
    top = random.randint(0, h - crop_size)
    img_cropped = img_noisy.crop((left, top, left + crop_size, top + crop_size))

    out_path = image_path.with_name(image_path.stem + "_aug.jpg")
    img_cropped.save(out_path)
    return out_path

# Apply to rendered images (skip already-augmented ones)
rendered_files = sorted(
    p for p in target_root.rglob("*.jpg") if not p.stem.endswith("_aug")
)
augmented_files = [augment_image(p) for p in rendered_files]

print("\nSample original images:")
for f in rendered_files[:8]:
    display(Image.open(str(f)))

print("\nSample augmented images:")
for f in augmented_files[:8]:
    display(Image.open(str(f)))
